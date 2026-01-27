from circuit import Component, Circuit


class Register(Component):
    """Holds a static value (TDDR, PRD)."""

    def __init__(self, name, value):
        super().__init__(name, [], ["out"])
        self.signals["out"].value = value  # Pre-initialize
        self.signals["out"].next_value = value

    def evaluate(self):
        # Register output is constant unless manually poked,
        # so we just enforce the current state.
        self["out"].set_next(self["out"].value)


class LogicGate(Component):
    def __init__(self, name, op):
        super().__init__(name, ["A", "B"], ["out"])
        self.op = op

    def evaluate(self):
        self["out"].set_next(1 if self.op(self["A"].value, self["B"].value) else 0)


class NOT(Component):
    """Inverter for TCR.4 logic."""

    def __init__(self, name):
        super().__init__(name, ["in"], ["out"])

    def evaluate(self):
        self["out"].set_next(1 - self["in"].value)

class LED(Component):
    """Inverter for TCR.4 logic."""

    def __init__(self, name):
        super().__init__(name, ["in"], ["out", "state"])

    def evaluate(self):
        self['state'].set_next(self['in'].value > self['out'].value)


class OR(Component):
    def __init__(self, name):
        super().__init__(name, ["A", "B"], ["out"])

    def evaluate(self):
        self["out"].set_next(1 if (self["A"].value or self["B"].value) else 0)


class AND(Component):
    def __init__(self, name):
        super().__init__(name, ["A", "B"], ["out"])

    def evaluate(self):
        self["out"].set_next(1 if (self["A"].value and self["B"].value) else 0)


class CPUTimerCounter(Component):
    """
    Generic Down-Counter for both Prescaler and Main Timer.
    - 'clk' input is the trigger (SYSCLK for Prescaler, Borrow for Main).
    - 'load' is the async/sync load signal.
    - 'borrow' is the output pulse when count == 0.
    """

    def __init__(self, name, start_val=0):
        super().__init__(name, ["clk", "load", "din"], ["out", "borrow"])
        self.state = start_val
        self.prev_clk = 0

    def evaluate(self):
        clk = self["clk"].value
        load = self["load"].value
        din = self["din"].value

        # Rising Edge Detection
        if self.prev_clk == 0 and clk == 1:
            if load:
                self.state = din
            else:
                if self.state == 0:
                    # In hardware, this usually wraps to Max,
                    # but here the load logic handles the reload via feedback.
                    # We'll wrap to 0xFFFF (simplification) or just decrement.
                    self.state = 0xFFFF
                else:
                    self.state -= 1

        self.prev_clk = clk

        # Output Logic
        self["out"].set_next(self.state)
        # Borrow is active High when state is 0.
        # This drives the feedback loop to reload on the NEXT clock.
        self["borrow"].set_next(1 if self.state == 0 else 0)


class SystemClock(Component):
    def __init__(self, name, period=2):
        super().__init__(name, [], ["clk"])
        self.period = period

    def evaluate(self):
        self["clk"].set_next(1 - self["clk"].value)

    def next_events(self, t):
        return [(t + self.period // 2, self.name)]


class Mux(Component):
    """Generic N-way Multiplexer."""

    def __init__(self, name: str, num_inputs: int):
        inputs = [f"in{i}" for i in range(num_inputs)] + ["sel"]
        super().__init__(name, inputs, ["out"])

    def evaluate(self):
        sel = self["sel"].value
        input_key = f"in{sel}"
        if input_key in self.signals:
            self["out"].set_next(self[input_key].value)


class InputQualifier(Component):
    """
    Implements Sync, 3-sample, 6-sample, and Async modes.
    Mode 00: Sync (Latched by SYSCLK)
    Mode 01: 3-sample
    Mode 10: 6-sample
    Mode 11: Async (Pass-through)
    """

    def __init__(self, name: str):
        super().__init__(name, ["in", "mode", "clk"], ["out"])
        self.history = []

    def evaluate(self):
        mode = self["mode"].value
        val = self["in"].value

        if mode == 3:  # Async
            self["out"].set_next(val)
        elif mode == 0:  # Sync (Simplified for this event model)
            self["out"].set_next(val)
        else:  # Sampling modes
            samples = 3 if mode == 1 else 6
            self.history.append(val)
            if len(self.history) > samples:
                self.history.pop(0)

            # If all samples in history are identical, update output
            if len(self.history) == samples and all(x == self.history[0] for x in self.history):
                self["out"].set_next(self.history[0])


class GPIODataLogic(Component):
    """Handles GPySET, GPyCLEAR, GPyTOGGLE, and GPyDAT(W) logic."""

    def __init__(self, name: str):
        inputs = ["set", "clear", "toggle", "dat_w", "master_sel"]
        super().__init__(name, inputs, ["out"])
        self.current_state = 0

    def evaluate(self):
        if self["set"].value:
            self.current_state = 1
        elif self["clear"].value:
            self.current_state = 0
        elif self["toggle"].value:
            self.current_state = 1 - self.current_state
        else:
            # Note: In real HW, this is controlled by the CPU write bus
            pass

        self["out"].set_next(self.current_state)


def demo():
    c = Circuit()

    # 1. Instantiate Components
    # Inputs / Registers
    clk_gen = SystemClock("SYSCLKOUT", period=2)
    reg_tddr = Register("TDDR", value=2)  # Prescaler Reload Value (Count 2, 1, 0)
    reg_prd = Register("PRD", value=5)  # Main Reload Value (Count 5..0)

    # Logic Gates
    gate_or_reset = OR("OR_Reset")  # Top Left OR (Reset | Reload)
    gate_or_pre = OR("OR_PreLoad")  # Feeds Prescaler Load
    gate_or_main = OR("OR_MainLoad")  # Feeds Main Timer Load
    gate_not_tcr = NOT("NOT_TCR")  # Inverts TCR.4
    gate_and_clk = AND("AND_Clk")  # Gates the SYSCLK

    # Counters
    cnt_pre = CPUTimerCounter("Prescaler")
    cnt_main = CPUTimerCounter("MainTimer")

    # Add to circuit
    for comp in [clk_gen, reg_tddr, reg_prd, gate_or_reset, gate_or_pre,
                 gate_or_main, gate_not_tcr, gate_and_clk, cnt_pre, cnt_main]:
        c.add(comp)

    # 2. Wiring (Based on Figure 3-10)

    # Clock Gating Logic: SYSCLK + TCR.4 -> Prescaler Clk
    c.connect("SYSCLKOUT", "clk", "AND_Clk", "A")
    c.connect("NOT_TCR", "out", "AND_Clk", "B")
    c.connect("AND_Clk", "out", "Prescaler", "clk")  # Gated Clock

    # Reset/Reload Logic
    # Note: We will manually poke inputs to OR_Reset, so no source connect needed for its inputs yet
    c.connect("OR_Reset", "out", "OR_PreLoad", "A")
    c.connect("OR_Reset", "out", "OR_MainLoad", "A")

    # Feedback Loops (The Red Arrows in your diagram)
    # Prescaler Borrow -> OR_PreLoad -> Prescaler Load
    c.connect("Prescaler", "borrow", "OR_PreLoad", "B")
    c.connect("OR_PreLoad", "out", "Prescaler", "load")

    # Prescaler Borrow -> Main Timer Clock (The cascade)
    c.connect("Prescaler", "borrow", "MainTimer", "clk")

    # Main Borrow -> OR_MainLoad -> Main Timer Load
    c.connect("MainTimer", "borrow", "OR_MainLoad", "B")
    c.connect("OR_MainLoad", "out", "MainTimer", "load")

    # Data Paths
    c.connect("TDDR", "out", "Prescaler", "din")
    c.connect("PRD", "out", "MainTimer", "din")

    # 3. Simulation Trace

    # Initialize Static Inputs
    c.poke("OR_Reset", "A", 0)  # Reset line
    c.poke("OR_Reset", "B", 0)  # Timer Reload line
    c.poke("NOT_TCR", "in", 0)  # TCR.4 = 0 (Start Timer). NOT(0) = 1.

    # Start Clock
    c.schedule("SYSCLKOUT", 0)

    # FORCE RESET TO INITIALIZE
    # We pulse the Reset line to load the Registers into the Counters
    print(">>> SYSTEM RESET <<<")
    c.poke("OR_Reset", "A", 1)
    c.run(steps=5)  # Allow propagate
    c.poke("OR_Reset", "A", 0)

    print("\n>>> STARTING TIMER <<<")
    print(f"{'Time':<5} | {'SysClk':<6} | {'Pre':<3} | {'P_Bor':<5} | {'Main':<4} | {'M_Bor':<5} | {'Event'}")
    print("-" * 65)

    # Run Simulation
    # Prescaler is set to 2. It counts: 2 -> 1 -> 0 (Borrow) -> Reload 2
    # Main is set to 5. It decrements every time Pre hits 0.

    for _ in range(25):
        c.run(steps=2)  # Run one full clock cycle (Low-High)

        # Capture signal values for printing
        sys_clk = clk_gen['clk'].value
        p_val = cnt_pre['out'].value
        p_bor = cnt_pre['borrow'].value
        m_val = cnt_main['out'].value
        m_bor = cnt_main['borrow'].value

        event_msg = ""
        if p_bor: event_msg += "[Pre Underflow] "
        if m_bor: event_msg += "[MAIN INTERRUPT]"

        print(f"{c.time:<5} | {sys_clk:<6} | {p_val:<3} | {p_bor:<5} | {m_val:<4} | {m_bor:<5} | {event_msg}")


def create_gpio_logic():
    c = Circuit()

    # 1. Inputs/Registers (Static or Poked)
    c.add(Register("GPyPUD", 1))  # Pull-up disable
    c.add(Register("GPyINV", 0))  # Input Inversion
    c.add(Register("GPyQSEL", 0))  # Qualification Selection
    c.add(Register("GPyDIR", 0))  # Direction (0=In, 1=Out)
    c.add(Register("GPyODR", 0))  # Open Drain
    c.add(SystemClock("SYSCLK", period=2))

    # 2. Input Path Logic
    # Inverter Mux (GPyINV)
    c.add(NOT("Inverter"))
    c.add(Mux("InvMux", 2))

    # Qualification
    c.add(InputQualifier("Qualifier"))

    # Input XBAR / Peripheral Mux (Output to Peripherals)
    c.add(Mux("PeripheralInMux", 16))

    # 3. Output Path Logic
    c.add(GPIODataLogic("DataLogic"))
    c.add(Mux("PeripheralOutMux", 16))

    # Open Drain and Enable Logic
    c.add(AND("EnableLogic"))

    # 4. Connections (Simulating the Wiring)
    # Input side
    c.connect("Inverter", "out", "InvMux", "in1")
    # (Simplified: Pin input goes to InvMux in0 and Inverter in)

    c.connect("InvMux", "out", "Qualifier", "in")
    c.connect("GPyQSEL", "out", "Qualifier", "mode")
    c.connect("SYSCLK", "clk", "Qualifier", "clk")

    c.connect("Qualifier", "out", "PeripheralInMux", "in0")  # Example: Routing to Peripheral A

    return c

def demo2():

    # Instantiate and Run
    gpio_circuit = create_gpio_logic()
    gpio_circuit.poke("InvMux", "in0", 1)  # Simulate pin high

    for i in range(20):
        gpio_circuit.run(steps=1)

        print(f"Time step {i}")
        print(gpio_circuit)

if __name__ == '__main__':
    # demo()
    demo2()
