import heapq
from collections import defaultdict

type Key = tuple[str, str]

DELAY = 1


class Signal:
    def __init__(self, value: int = 0, last_value: int = 0):
        self.value = value
        self.last_value = last_value

    def update(self, value: int) -> bool:
        changed = self.value != value

        self.last_value = self.value
        self.value = value

        return changed

    def __repr__(self) -> str:
        return f"{self.last_value}->{self.value}"

    def changed(self):
        return self.value != self.last_value


class Component:
    def __init__(self, name: str, inputs: list[str], outputs: list[str]):
        self.name = name

        self.inputs = inputs
        self.outputs = outputs

        self.signals: dict[str, Signal] = dict()

        for inp in inputs:
            self.signals[inp] = Signal()

        for out in outputs:
            self.signals[out] = Signal()

    def __getitem__(self, item) -> Signal:
        return self.signals[item]

    def __setitem__(self, key, value):
        self.signals[key] = value

    def evaluate(self) -> bool:
        pass

    def step_visuals(self) -> None:
        pass

    def changed(self) -> list[Key]:
        return [(self.name, n) for n, s in self.signals.items() if s.changed()]

    def __repr__(self) -> str:
        signals: list[str] = []

        for n, signal in self.signals.items():
            signals.append(f"    {n}: {signal}")

        lines = [self.name] + signals

        return "\n".join(lines)


class Event:
    def __init__(self, time: int, component: str):
        self.time = time
        self.component = component


class Circuit:
    def __init__(self):
        self.components: dict[str, Component] = dict()

        self.graph: dict[Key, list[Key]] = defaultdict(list)

        self.events: list[tuple[int, str]] = []

        self.t = 0

    def add_component(self, component: Component):

        self.components[component.name] = component

    def connect(self, from_comp: Key, to_comp: Key):
        self.graph[from_comp].append(to_comp)

    def run(self, inputs: list[tuple[Key, bool]]):
        self.t += DELAY

        for (comp, inp), val in inputs:
            if self.components[comp][inp].update(val):
                self.events.append((self.t, comp))

        while len(self.events) > 0:
            self.step()

    def step(self) -> None:
        if len(self.events) == 0:
            return

        t = self.events[0][0]

        active = set()

        i = 0
        while i < len(self.events) and self.events[i][0] == t:
            active.add(heapq.heappop(self.events)[1])

        changed_set: set[Key] = set()

        for c in active:
            comp = self.components[c]

            if comp.evaluate():
                comp.step_visuals()

                changed_signals = comp.changed()

                changed_set.update(changed_signals)

                self.update_signals(changed_signals)

        for c in changed_set:
            for n, signal in self.graph[c]:
                heapq.heappush(self.events, (t + DELAY, n))

    def update_signals(self, val: list[Key]):
        for v in val:
            self.update_signal(v)

    def update_signal(self, inp: Key):
        state = self.components[inp[0]][inp[1]]

        for n in self.graph[inp]:
            self.components[n[0]][inp[0]] = state

    def __repr__(self):
        components: list[str] = []

        components.append("Values")

        for component in self.components.values():
            components.append(str(component))

        return "\n".join(components)


class OR(Component):
    def __init__(self, name: str):
        super().__init__(name, ["A", "B"], ["out"])

    def evaluate(self) -> bool:
        a = self['A']
        b = self['B']

        out_val = a.value or b.value

        out = self['out']

        return out.update(out_val)


class AND(Component):
    def __init__(self, name: str):
        super().__init__(name, ["A", "B"], ["out"])

    def evaluate(self) -> bool:
        a = self['A']
        b = self['B']

        out_val = a.value and b.value

        out = self['out']

        return out.update(out_val)


if __name__ == '__main__':
    or_ = OR('A')
    and_ = AND('B')

    circuit = Circuit()

    circuit.add_component(or_)
    circuit.add_component(and_)

    circuit.connect(("A", "B"), ("B", "A"))

    print(circuit)

    circuit.run([
        (("A", "B"), True),
        (("B", "B"), True),
    ])

    print(circuit)
