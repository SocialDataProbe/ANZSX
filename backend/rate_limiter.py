import asyncio
import random
import time

UTILISATION = 0.8

# Start Usage Tier, per model: (requests/min, input tokens/min, output tokens/min)
START_TIER_LIMITS = {
    "fable": (1_000, 500_000, 100_000),
    "default": (1_000, 2_000_000, 400_000),
}

# TypeSafe (Jev), per second: (requests/s, input tokens/s)
JEV_LIMITS = (80, 100_000)

BURST_SECONDS = 10


def start_tier_limits(model: str) -> dict:
    key = "fable" if "fable" in model else "default"
    rpm, itpm, otpm = START_TIER_LIMITS[key]
    return {"rpm": rpm, "itpm": itpm, "otpm": otpm, "source": "Start tier default"}


def backoff_seconds(attempt: int, retry_after: float | None = None) -> float:
    if retry_after is not None:
        return min(60.0, retry_after) + random.uniform(0, 1)
    return min(60.0, 2 ** attempt * 2) * random.uniform(0.5, 1.0)


class TokenBucket:
    def __init__(self, rate_per_sec: float, burst_seconds: float = BURST_SECONDS):
        self.rate = rate_per_sec
        self.capacity = rate_per_sec * burst_seconds
        self.level = self.capacity
        self.burst_seconds = burst_seconds
        self.last = time.monotonic()

    def set_rate(self, rate_per_sec: float):
        self._refill()
        self.rate = rate_per_sec
        self.capacity = rate_per_sec * self.burst_seconds
        self.level = min(self.level, self.capacity)

    def _refill(self):
        now = time.monotonic()
        self.level = min(self.capacity, self.level + (now - self.last) * self.rate)
        self.last = now

    def wait_time(self, amount: float) -> float:
        self._refill()
        # Requests bigger than the bucket go through once it is full, leaving it in debt.
        needed = min(amount, self.capacity)
        if self.level >= needed:
            return 0.0
        return (needed - self.level) / self.rate

    def take(self, amount: float):
        self._refill()
        self.level -= amount

    def give_back(self, amount: float):
        self._refill()
        self.level = min(self.capacity, self.level + amount)


class RateLimiter:
    """Paces request starts to stay under request/input/output token limits.

    Charges an estimate at dispatch, then corrects it with actual usage.
    Concurrency ramps up gradually and halves after a 429.
    """

    def __init__(
        self,
        requests_per_sec: float,
        input_per_sec: float | None,
        output_per_sec: float | None = None,
        *,
        max_concurrency: int = 50,
        ramp_start: int = 10,
        ramp_seconds: float = 120.0,
    ):
        self.requests = TokenBucket(requests_per_sec * UTILISATION)
        self.input = TokenBucket(input_per_sec * UTILISATION) if input_per_sec else None
        self.output = TokenBucket(output_per_sec * UTILISATION) if output_per_sec else None
        self.max_concurrency = max(1, max_concurrency)
        self.ramp_start = min(ramp_start, self.max_concurrency)
        self.ramp_seconds = ramp_seconds
        self.started = time.monotonic()
        self.ceiling = float(self.max_concurrency)
        self.ceiling_updated = self.started
        self.paused_until = 0.0
        self.in_flight = 0
        self.min_interval = 1.0 / (requests_per_sec * UTILISATION)
        self.last_start = 0.0
        self._lock = asyncio.Lock()
        self._released = asyncio.Event()

    @classmethod
    def for_model(cls, model: str, max_concurrency: int) -> "RateLimiter":
        lim = start_tier_limits(model)
        return cls(
            lim["rpm"] / 60, lim["itpm"] / 60, lim["otpm"] / 60,
            max_concurrency=max_concurrency,
        )

    @classmethod
    def for_jev(cls, max_concurrency: int) -> "RateLimiter":
        rps, tps = JEV_LIMITS
        return cls(rps, tps, None, max_concurrency=max_concurrency, ramp_seconds=30.0)

    def limits(self) -> dict:
        def per_min(bucket):
            return int(round(bucket.rate / UTILISATION * 60)) if bucket else None
        return {
            "rpm": per_min(self.requests),
            "itpm": per_min(self.input),
            "otpm": per_min(self.output),
        }

    def update_from_headers(self, headers) -> bool:
        """Adopt the real per-minute limits from anthropic-ratelimit-*-limit headers."""
        changed = False
        for name, bucket in (
            ("requests", self.requests),
            ("input-tokens", self.input),
            ("output-tokens", self.output),
        ):
            value = headers.get(f"anthropic-ratelimit-{name}-limit")
            if bucket is None or not value:
                continue
            try:
                rate = float(value) / 60 * UTILISATION
            except ValueError:
                continue
            if rate > 0 and abs(rate - bucket.rate) / bucket.rate > 0.01:
                bucket.set_rate(rate)
                if name == "requests":
                    self.min_interval = 1.0 / rate
                changed = True
        return changed

    def current_cap(self) -> int:
        now = time.monotonic()
        # Ceiling recovers by one slot every 2 s after a 429.
        if self.ceiling < self.max_concurrency:
            self.ceiling = min(
                self.max_concurrency,
                self.ceiling + (now - self.ceiling_updated) / 2.0,
            )
        self.ceiling_updated = now
        frac = min(1.0, (now - self.started) / self.ramp_seconds) if self.ramp_seconds else 1.0
        ramp = self.ramp_start + (self.max_concurrency - self.ramp_start) * frac
        return max(1, int(min(ramp, self.ceiling)))

    async def acquire(self, est_input: float, est_output: float = 0.0) -> tuple[float, float]:
        async with self._lock:
            while True:
                now = time.monotonic()
                waits = [
                    self.paused_until - now,
                    self.last_start + self.min_interval - now,
                    self.requests.wait_time(1),
                ]
                if self.input:
                    waits.append(self.input.wait_time(est_input))
                if self.output:
                    waits.append(self.output.wait_time(est_output))
                wait = max(waits)
                if self.in_flight >= self.current_cap():
                    self._released.clear()
                    try:
                        await asyncio.wait_for(self._released.wait(), timeout=max(wait, 1.0))
                    except asyncio.TimeoutError:
                        pass
                    continue
                if wait > 0:
                    await asyncio.sleep(wait)
                    continue
                self.requests.take(1)
                if self.input:
                    self.input.take(est_input)
                if self.output:
                    self.output.take(est_output)
                self.last_start = time.monotonic()
                self.in_flight += 1
                return est_input, est_output

    def release(
        self,
        ticket: tuple[float, float],
        actual_input: float | None,
        actual_output: float | None,
    ):
        """Correct the charge with actual usage. None means the request didn't count."""
        est_input, est_output = ticket
        self.in_flight -= 1
        if self.input:
            self.input.give_back(est_input - (actual_input or 0))
        if self.output:
            self.output.give_back(est_output - (actual_output or 0))
        if actual_input is None:
            self.requests.give_back(1)
        self._released.set()

    def on_rate_limited(self, retry_after: float | None):
        now = time.monotonic()
        pause = retry_after if retry_after is not None else 5.0
        self.paused_until = max(self.paused_until, now + pause)
        self.ceiling = max(1.0, min(self.ceiling, self.current_cap()) / 2)
        self.ceiling_updated = now
