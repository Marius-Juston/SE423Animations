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


class NOT(Component):
    """Inverter for TCR.4 logic."""

    def __init__(self, name):
        super().__init__(name, ["in"], ["out"])

    def evaluate(self):
        self["out"].set_next(1 - self["in"].value)


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

    def __init__(self, name):
        super().__init__(name, ["clk", "load", "din"], ["out", "borrow"])
        self.state = 0
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




if __name__ == "__main__":
    demo()

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


if __name__ == '__main__':
    demo()