"""
Рутковский А.М. 121703

Вариант 2: сеть Хопфилда с непрерывным состоянием и дискретным временем в синхронном режиме.
 - метод дельта-проекций
 - функция активации - гиперболический тангенс

# Ссылки на источники:
# https://numpy.org/doc/2.1/reference/index.html
# https://github.com/iit-students-charity/mrz5-2/blob/master/main.py

"""
import random

import numpy as np


class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns, nu=0.7, error=1e-6, max_iterations=10000):
        patterns = self._remove_orthogonal_and_inverted_patterns(patterns)
        self.patterns = patterns

        epochs = 0

        for _ in range(max_iterations):
            epochs += 1

            old_weights = self.weights.copy()

            for pattern in patterns:
                x_t = np.array(pattern).reshape(-1, 1)
                activation = self.activation_function(self.weights @ x_t)
                correction = x_t - activation
                delta_w = correction @ x_t.T
                self.weights += (nu / self.size) * delta_w
                np.fill_diagonal(self.weights, 0)

            if np.abs(old_weights - self.weights).sum() < error:
                break

        np.fill_diagonal(self.weights, 0)

    def activation_function(self, x):
        return np.tanh(x)

    def update(self, state):
        res = self.activation_function(np.dot(self.weights, state))
        return res

    def predict(self, input_pattern, max_iterations=100_000, tolerance=1e-5):
        states = [np.matrix(input_pattern.copy())] * 4

        state = np.array(input_pattern)
        for iteration in range(max_iterations):
            new_state = self.update(state)

            states.append(new_state)
            states.pop(0)

            if (
                iteration >= 3
                and np.abs(states[0] - states[2]).max() < tolerance
                and np.abs(states[1] - states[3]).max() < tolerance
            ):
                idx, is_inverted = self._get_original_idx(new_state)
                print(f"Сеть достигла состояния релаксации на {
                      iteration + 1} итерации")
                return new_state, idx, is_inverted
            state = new_state
        return state, None, None

    def _get_original_idx(self, state):
        for idx, image in enumerate(self.patterns):
            if np.abs(image - state).max() < 1e-2:
                return idx, False
            if np.abs((np.array(image) * -1) - state).max() < 1e-2:
                return idx, True
        return None, None

    @staticmethod
    def _remove_orthogonal_and_inverted_patterns(patterns, tolerance=1e-5):
        filtered_patterns = []
        for i, pattern in enumerate(patterns):
            is_valid = True
            for j, other_pattern in enumerate(patterns):
                if i == j:
                    continue

                dot_product = np.dot(pattern, other_pattern)
                if abs(dot_product) < tolerance:
                    print(
                        f"Паттерн {i + 1} ортогонален паттерну {j + 1}. Удалён.")
                    is_valid = False
                    break

                if np.array_equal(pattern, -np.array(other_pattern)):
                    print(
                        f"Паттерн {i + 1} является инверсией паттерна {j + 1}. Удалён.")
                    is_valid = False
                    break
            if is_valid:
                filtered_patterns.append(pattern)
        return filtered_patterns


def display_console_pattern(pattern, size=4, name: int | str = "Pattern"):
    translator = {-1: " ", 1: "⌘", 0: "▒"}
    print(f"Паттерн: {name}")
    print("+" + "-" * size + "+")
    for i in range(size):
        row = pattern[i * size:(i + 1) * size]
        print("|" + "".join(translator[val] for val in row) + "|")
    print("+" + "-" * size + "+\n")


def add_noise_to_pattern(pattern, noise_level=0.2):
    noisy_pattern = pattern.copy()
    for i in range(len(noisy_pattern)):
        if random.random() < noise_level:
            if random.random() < 0.5:
                noisy_pattern[i] = 0
            else:
                noisy_pattern[i] *= -1
    return noisy_pattern


if __name__ == "__main__":
    patterns = [
        # Буква L
        [1, -1, -1, -1,
         1, -1, -1, -1,
         1, -1, -1, -1,
         1,  1,  1, 1],

        # Буква T
        [1,  1,  1,  1,
         -1, -1,  1, -1,
         -1, -1,  1, -1,
         -1, -1,  1, -1],

        # Буква U
        [1, -1,  -1, 1,
         1, -1,  -1, 1,
         1, -1,  -1, 1,
         1,  1,   1, 1],

        # # Буква O
        [1,  1,  1, 1,
         1, -1,  -1, 1,
         1, -1,  -1, 1,
         1,  1,  1, 1],

        # Буква X
        [1, -1, -1,  1,
         -1,  1,  1, -1,
         -1,  1,  1, -1,
         1, -1, -1,  1]
    ]

    network = HopfieldNetwork(size=len(patterns[0]))
    network.train(patterns)

    for idx, pattern in enumerate(patterns, 1):
        display_console_pattern(pattern, name=idx)

    noisy_input = add_noise_to_pattern(patterns[1])
    display_console_pattern(noisy_input, name="Входной образ")

    result, idx, is_inverted = network.predict(noisy_input)

    print("Результат работы сети:", result)
    if idx is not None:
        print(f"Распознан как образ номер {
              idx + 1}. {"Входной образ был инверсией" if is_inverted else ""}")
    else:
        print("Сеть не смогла распознать входной образ")

    display_console_pattern(np.sign(result), name="result")
