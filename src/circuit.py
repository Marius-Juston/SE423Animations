import heapq
from collections import defaultdict

type Key = tuple[str, str]
type Value = int | bool

DELAY = 1


class Signal:
    def __init__(self, value: Value = 0):
        self.value = value
        self.next_value = value

    def set_next(self, value: Value):
        self.next_value = value

    def commit(self):
        changed = self.value != self.next_value
        self.value = self.next_value
        return changed

    def __repr__(self):
        return str(self.value)


class Component:
    def __init__(self, name: str, inputs: list[str], outputs: list[str]):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs

        self.signals = {s: Signal() for s in inputs + outputs}

    def evaluate(self):
        raise NotImplementedError

    def commit(self):
        changed = False
        for o in self.outputs:
            changed |= self.signals[o].commit()
        return changed

    def next_events(self, t: int):
        return []

    def __getitem__(self, key: str):
        return self.signals[key]

    def __repr__(self):
        lines = [self.name]
        for k, v in self.signals.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)


class Circuit:
    def __init__(self, delay=DELAY):
        self.delay = delay
        self.components = {}
        self.deps = defaultdict(set)  # component → downstream components
        self.events = []
        self.time = 0
        self.scheduled = set()

    def add(self, c: Component):
        self.components[c.name] = c

    def schedule(self, comp: str, t: int):
        if (comp, t) not in self.scheduled:
            heapq.heappush(self.events, (t, comp))
            self.scheduled.add((comp, t))

    def connect(self, src_comp: str, src_pin: str, dst_comp: str, dst_pin: str):

        src = self.components[src_comp]
        dst = self.components[dst_comp]

        dst.signals[dst_pin] = src.signals[src_pin]
        self.deps[src_comp].add(dst_comp)

    def poke(self, comp: str, signal: str, value: Value):
        c = self.components[comp]
        c[signal].set_next(value)
        c[signal].commit()
        self.schedule(comp, self.time)

    def run(self, steps: int = 100) -> bool:
        updated_data = False

        while self.events and steps > 0:
            t, _ = self.events[0]
            self.time = t

            active = set()
            while self.events and self.events[0][0] == t:
                _, c = heapq.heappop(self.events)
                self.scheduled.discard((c, t))
                active.add(c)

            # Phase 1: evaluate
            for name in active:
                self.components[name].evaluate()

            # Phase 2: commit
            changed = set()
            for name in active:
                if self.components[name].commit():
                    changed.add(name)

            # Phase 3: schedule dependents
            for c in changed:
                for n in self.deps[c]:
                    updated_data = True
                    self.schedule(n, t + self.delay)

            # Phase 4: Autonomous components
            for name in active:
                comp = self.components[name]
                for (nt, target) in comp.next_events(t):
                    self.schedule(target, nt)

            steps -= 1

        return updated_data

    def __repr__(self):
        components = [f"Num components: {len(self.components)}"]

        for com in self.components.values():
            components.append(f"{com}")

        return "\n".join(components)


class AND(Component):
    def __init__(self, name: str):
        super().__init__(name, ["A", "B"], ["out"])

    def evaluate(self):
        self["out"].set_next(self["A"].value and self["B"].value)


class OR(Component):
    def __init__(self, name: str):
        super().__init__(name, ["A", "B"], ["out"])

    def evaluate(self):
        self["out"].set_next(self["A"].value or self["B"].value)


class DownCounter(Component):
    def __init__(self, name: str, bits: int = 4):
        super().__init__(name, ["clk", "load", "din"], ["out"])
        self.bits = bits
        self.state = 0
        self.prev_clk = 0

    def evaluate(self):
        clk = self["clk"].value

        # rising edge
        if self.prev_clk == 0 and clk == 1:
            if self["load"].value:
                self.state = self["din"].value
            else:
                self.state = (self.state - 1) % (1 << self.bits)

        self.prev_clk = clk
        self["out"].set_next(self.state)


class Clock(Component):
    def __init__(self, name: str, period: int = 2):
        super().__init__(name, [], ["clk"])
        self.period = period

    def evaluate(self):
        self["clk"].set_next(1 - self["clk"].value)

    def next_events(self, t):
        return [(t + self.period // 2, self.name)]


def demo1():
    c = Circuit()

    period = 2

    clk = Clock("clk", period=period)
    cnt = DownCounter("cnt", bits=3)

    c.add(clk)
    c.add(cnt)

    c.connect("clk", "clk", "cnt", "clk")

    # initialize
    c.poke("cnt", "load", 1)
    c.poke("cnt", "din", 5)

    # start the clock explicitly
    c.schedule("clk", 0)

    c.run(steps=period)

    c.poke("cnt", "load", 0)

    for _ in range(15):
        if not c.run(steps=1):
            break
        print(f"time={c.time}, clk={clk['clk'].value}, cnt={cnt['out'].value}")


if __name__ == "__main__":
    demo1()
