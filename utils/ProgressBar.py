class ProgressBar:
    def __init__(self, max_value, modulo_threshold):
        self.max_value = max_value
        self.modulo_threshold = modulo_threshold

    def update(self, i):
        last = (i + self.modulo_threshold - 1) > self.max_value
        if (i % self.modulo_threshold) == 0 or last:
            max_print = 40
            i += 1
            n = int((i / self.max_value) * 100)
            n_symbols = int(max_print * (n / 100))
            spaces = "-" * int(max_print - n_symbols)
            i = self.max_value if (i + 1) == self.max_value else i
            n_str = str(i) + "/" + str(self.max_value) + "]"
            print("\r", "[" + "#" * n_symbols + spaces + "] [" + str(n) + " %] [" + n_str, end=" ")
