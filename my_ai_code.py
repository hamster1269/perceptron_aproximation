import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import time
import matplotlib.pyplot as plt

# Важно: используем tqdm.notebook для Jupyter/Colab
from tqdm.notebook import tqdm
import collections  # Для словаря функций

# ---- Конфигурация ----
N_SAMPLES = 1000  # Количество обучающих точек
N_TEST_SAMPLES = 300  # Количество тестовых точек (для отрисовки)
# X_MIN, X_MAX теперь будут определяться для каждой функции индивидуально
NUM_EPOCHS = 2000  # Количество эпох обучения

# Перемещаем данные на GPU, если доступно
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
print(f"Number of training epochs: {NUM_EPOCHS}")

# ---- Целевые функции и их диапазоны аппроксимации ----
# Теперь значение словаря - это под-словарь с функцией и ее диапазоном
target_functions = collections.OrderedDict(
    {
        "sin(x) + sin(2x)": {
            "func": lambda x: torch.sin(x) + torch.sin(2.0 * x),
            "range": (-math.pi, math.pi),
        },
        "cos(3x)": {
            "func": lambda x: torch.cos(3.0 * x),
            "range": (-4.0, 4.0),  # Другой диапазон
        },
        "x^2 / 5": {"func": lambda x: x**2 / 5.0, "range": (-3.0, 3.0)},
        "tanh(2x)": {"func": lambda x: torch.tanh(2 * x), "range": (-2.5, 2.5)},
        "exp(-x^2)": {
            "func": lambda x: torch.exp(-(x**2)),  # Гауссиана
            "range": (-3.5, 3.5),
        },
        "abs(x) - 1": {
            "func": lambda x: torch.abs(x) - 1.0,  # Негладкая
            "range": (-4.5, 4.5),
        },
        "x * sin(x)": {
            "func": lambda x: x * torch.sin(x),  # Затухающие колебания
            "range": (-2 * math.pi, 2 * math.pi),  # Более широкий диапазон
        },
        "0.5*x + sin(2x)": {
            "func": lambda x: 0.5 * x
            + torch.sin(2.0 * x),  # Линейный тренд + колебания
            "range": (-5.0, 5.0),
        },
        "log(x^2 + 1)": {
            "func": lambda x: torch.log(x**2 + 1.0),  # Логарифмическая, симметричная
            "range": (-6.0, 6.0),
        },
        "max(0, sin(pi*x))": {
            "func": lambda x: torch.relu(torch.sin(math.pi * x)),  # Полуволна синусоиды
            "range": (-1.5, 2.5),  # Несимметричный диапазон
        },
        "step(x) (approx)": {
            "func": lambda x: torch.tanh(10 * x),  # Аппроксимация ступеньки
            "range": (-2.0, 2.0),
        },
        # Ваша функция с ограничением диапазона из-за sqrt и log2
        "log2(x^2)+sqrt(x)": {
            # Добавил relu и смещение для стабильности sqrt и log2 около нуля
            "func": lambda x: torch.log2(x**2 + 1e-6)
            + torch.sqrt(torch.relu(x) + 1e-6),
            "range": (0.1, 5.0),  # Строго положительный диапазон
        },
    }
)
print(
    f"\nTarget functions and their ranges: { {k: v['range'] for k, v in target_functions.items()} }"
)


# ---- Определение архитектур нейронных сетей ----
# (Класс SimpleNN остается без изменений)
class SimpleNN(nn.Module):
    def __init__(
        self,
        input_size=1,
        output_size=1,
        hidden_layers=1,
        hidden_size=10,
        activation_fn=nn.ReLU(),
    ):
        super(SimpleNN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(activation_fn)
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation_fn)
        layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


network_configs = {
    "Shallow_Wide_ReLU": {
        "hidden_layers": 1,
        "hidden_size": 128,
        "activation_fn": nn.ReLU(),
        "name": "Shallow Wide (ReLU)",
    },
    "Deep_Narrow_ReLU": {
        "hidden_layers": 4,
        "hidden_size": 16,
        "activation_fn": nn.ReLU(),
        "name": "Deep Narrow (ReLU)",
    },
    "Mid_Tanh": {
        "hidden_layers": 2,
        "hidden_size": 32,
        "activation_fn": nn.Tanh(),
        "name": "Mid Depth/Width (Tanh)",
    },
    "Mid_Sigmoid": {
        "hidden_layers": 2,
        "hidden_size": 32,
        "activation_fn": nn.Sigmoid(),
        "name": "Mid Depth/Width (Sigmoid)",
    },
    "Deep_Wide_ReLU": {
        "hidden_layers": 3,
        "hidden_size": 64,
        "activation_fn": nn.ReLU(),
        "name": "Deep Wide (ReLU)",
    },
}


# ---- Определение оптимизаторов ----
# (Функция get_optimizer и optimizer_configs остаются без изменений)
def get_optimizer(model_params, config):
    name = config["name"]
    lr = config.get("lr", 0.01)
    if name == "SGD":
        return optim.SGD(model_params, lr=lr)
    elif name == "SGD_Momentum":
        momentum = config.get("momentum", 0.9)
        return optim.SGD(model_params, lr=lr, momentum=momentum)
    elif name == "Adam":
        lr = config.get("lr", 0.005)  # Adam часто требует меньший LR
        return optim.Adam(model_params, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


optimizer_configs = [
    {"name": "SGD", "lr": 0.01},
    {"name": "SGD_Momentum", "lr": 0.01, "momentum": 0.9},
    {"name": "Adam", "lr": 0.005},
]


# ---- Функция обучения и оценки ----
# (Функция train_and_evaluate остается без изменений)
def train_and_evaluate(
    network_config,
    optimizer_config,
    x_train,
    y_train,
    x_test,
    y_test,
    num_epochs=500,
    device="cpu",
):
    """Обучает модель и возвращает метрики."""
    model = SimpleNN(
        hidden_layers=network_config["hidden_layers"],
        hidden_size=network_config["hidden_size"],
        activation_fn=network_config["activation_fn"],
    ).to(device)
    optimizer = get_optimizer(model.parameters(), optimizer_config)
    criterion = nn.MSELoss()

    start_train_time = time.time()
    losses = []
    for epoch in tqdm(
        range(num_epochs),
        desc=f"Training {network_config['name']}+{optimizer_config['name']}",
        leave=False,
    ):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    end_train_time = time.time()
    training_time = end_train_time - start_train_time

    model.eval()
    with torch.no_grad():
        y_pred_test = model(x_test)
        test_mse = criterion(y_pred_test, y_test).item()

    single_x_test = x_test[0:1].to(device)
    inference_times = []
    n_inference_runs = 100
    with torch.no_grad():  # Прогрев
        _ = model(single_x_test)
    inf_start = time.time()
    for _ in range(n_inference_runs):
        with torch.no_grad():
            _ = model(single_x_test)
    inf_end = time.time()
    avg_inference_time = (inf_end - inf_start) / n_inference_runs

    return {
        "network_name": network_config["name"],
        "optimizer_name": optimizer_config["name"],
        "training_time": training_time,
        "final_mse_test": test_mse,
        "avg_inference_time": avg_inference_time,
        "losses": losses,
        "predictions": y_pred_test.cpu().numpy(),
    }


# ---- Основной цикл по функциям ----
plt.style.use("seaborn-v0_8-darkgrid")

# Итерация по словарю функций
for func_name, func_data in target_functions.items():
    # Получаем функцию и ее уникальный диапазон
    target_func = func_data["func"]
    X_MIN, X_MAX = func_data["range"]

    print(
        f"\n{'='*10} Processing Function: {func_name} on range [{X_MIN:.3f}, {X_MAX:.3f}] {'='*10}"
    )

    # 1. Генерация данных для текущей функции в ее диапазоне
    x_train_np = np.random.uniform(X_MIN, X_MAX, N_SAMPLES)
    x_train = torch.tensor(x_train_np, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    y_train = target_func(x_train).to(DEVICE)

    x_test_np = np.linspace(X_MIN, X_MAX, N_TEST_SAMPLES)
    x_test = torch.tensor(x_test_np, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    y_test = target_func(x_test).to(DEVICE)

    # 2. Запуск экспериментов для текущей функции
    results_current_func = []
    print("Running network/optimizer combinations...")
    combination_iterator = tqdm(
        total=len(network_configs) * len(optimizer_configs),
        desc=f"Combinations for {func_name}",
        leave=True,
    )
    for net_key, net_conf in network_configs.items():
        for opt_conf in optimizer_configs:
            result = train_and_evaluate(
                net_conf,
                opt_conf,
                x_train,
                y_train,
                x_test,
                y_test,
                num_epochs=NUM_EPOCHS,
                device=DEVICE,
            )
            results_current_func.append(result)
            combination_iterator.update(1)
    combination_iterator.close()

    # 3. Построение Графика 1 для текущей функции
    plt.figure(figsize=(12, 7))
    sorted_results = sorted(results_current_func, key=lambda r: r["losses"][-1])

    print(
        f"\n--- Results for Function '{func_name}' (Sorted by Final Training MSE) ---"
    )
    for i, res in enumerate(sorted_results):
        label = f"{res['network_name']} + {res['optimizer_name']}"
        final_train_mse = res["losses"][-1]
        inf_time_us = res["avg_inference_time"] * 1e6
        train_time_s = res["training_time"]
        full_label = f"{label} (Tr: {train_time_s:.1f}s, Inf: {inf_time_us:.1f}us)"
        print(
            f"{i+1}. {label}: Final Train MSE = {final_train_mse:.6e}, Train Time = {train_time_s:.1f}s, Inf Time = {inf_time_us:.1f} us, Final Test MSE = {res['final_mse_test']:.6e}"
        )
        plt.plot(res["losses"], label=full_label, alpha=0.8)

    plt.yscale("log")
    # В заголовок добавим и диапазон для наглядности
    plt.title(
        f'Function: "{func_name}" on [{X_MIN:.2f}, {X_MAX:.2f}]\nTraining Loss (MSE) vs. Epochs ({NUM_EPOCHS} Epochs)\n(Legend sorted by final training MSE)'
    )
    plt.xlabel("Epoch")
    plt.ylabel("Training MSE (log scale)")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0.0)
    plt.subplots_adjust(right=0.65)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.show()

print(f"\n{'='*10} All functions processed {'='*10}")
