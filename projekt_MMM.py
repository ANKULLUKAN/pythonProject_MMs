import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk


#model
def u_step(t, amplitude, duration):
    return amplitude if t < duration else 0

def u_triangle(t, amplitude, period):
    return amplitude * (2/period) * (period/2 - abs((t % period) - period/2))

def u_sine(t, amplitude, frequency):
    return amplitude * np.sin(2 * np.pi * frequency * t)

def model(x, t, u_func, M, b, k):
    x1, x2 = x
    u = u_func(t)
    dx1dt = x2
    dx2dt = (u - b * x2 - k * x1) / M
    return np.array([dx1dt, dx2dt])

def euler(x0, u_func, dt, T, M, b, k):
    t = np.arange(0, T, dt)
    x = np.zeros((len(t), len(x0)))
    x[0] = x0
    for i in range(1, len(t)):
        x[i] = x[i-1] + dt * model(x[i-1], t[i-1], u_func, M, b, k)
    return t, x

def rk4(x0, u_func, dt, T, M, b, k):
    t = np.arange(0, T, dt)
    x = np.zeros((len(t), len(x0)))
    x[0] = x0
    for i in range(1, len(t)):
        k1 = model(x[i-1], t[i-1], u_func, M, b, k)
        k2 = model(x[i-1] + dt/2*k1, t[i-1] + dt/2, u_func, M, b, k)
        k3 = model(x[i-1] + dt/2*k2, t[i-1] + dt/2, u_func, M, b, k)
        k4 = model(x[i-1] + dt*k3, t[i-1] + dt, u_func, M, b, k)
        x[i] = x[i-1] + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    return t, x


def run_simulation():
    try:
        M = float(entry_M.get())
        b = float(entry_b.get())
        k = float(entry_k.get())
        r = float(entry_r.get())
        dt = float(entry_dt.get())
        T = float(entry_T.get())
        fi_deg = float(entry_fi.get())
        signal = signal_type.get()

        fi_rad = fi_deg * np.pi / 180
        x0 = [-r * fi_rad, 0.0]

        if signal == "Skok":
            amp = float(param1.get())
            dur = float(param2.get())
            u_func = lambda t: u_step(t, amp, dur)
        elif signal == "Trójkąt":
            amp = float(param1.get())
            per = float(param2.get())
            u_func = lambda t: u_triangle(t, amp, per)
        else:  # Sinus
            amp = float(param1.get())
            freq = float(param2.get())
            u_func = lambda t: u_sine(t, amp, freq)


        t_euler, x_euler = euler(x0, u_func, dt, T, M, b, k)
        t_rk4, x_rk4 = rk4(x0, u_func, dt, T, M, b, k)
        u_values = np.array([u_func(ti) for ti in t_euler])


        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(t_euler, u_values, label='Wymuszenie u(t)', color='purple')
        plt.ylabel('u(t)')
        plt.title('Sygnał wejściowy')
        plt.grid()
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(t_euler, x_euler[:, 0], label='Euler  x(t)')
        plt.plot(t_rk4, x_rk4[:, 0], label='RK4  x(t)', linestyle='--')
        plt.ylabel('Pozycja x(t) [m]')
        plt.title('Pozycja układu')
        plt.grid()
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(t_euler, x_euler[:, 1], label='Euler  v(t)')
        plt.plot(t_rk4, x_rk4[:, 1], label='RK4  v(t)', linestyle='--')
        plt.ylabel('Prędkość v(t) [m/s]')
        plt.xlabel('Czas [s]')
        plt.title('Prędkość układu')
        plt.grid()
        plt.legend()

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Błąd:", e)


root = tk.Tk()
root.title("Symulacja układu ")

fields = {
    "Masa M (kg)": "1.0",
    "Tłumienie b (Ns/m)": "0.5",
    "Sprężystość k (N/m)": "2.0",
    "Promień r (m)": "0.1",
    "Krok czasowy dt (s)": "0.01",
    "Czas symulacji T (s)": "10",
    "Kąt początkowy θ [deg]": "30"
}

entries = {}
for i, (label, default) in enumerate(fields.items()):
    tk.Label(root, text=label).grid(row=i, column=0, sticky="e")
    ent = tk.Entry(root)
    ent.insert(0, default)
    ent.grid(row=i, column=1)
    entries[label] = ent

entry_M = entries["Masa M (kg)"]
entry_b = entries["Tłumienie b (Ns/m)"]
entry_k = entries["Sprężystość k (N/m)"]
entry_r = entries["Promień r (m)"]
entry_dt = entries["Krok czasowy dt (s)"]
entry_T = entries["Czas symulacji T (s)"]
entry_fi = entries["Kąt początkowy θ [deg]"]


signal_type = ttk.Combobox(root, values=["Skok", "Trójkąt", "Sinus"])
signal_type.current(0)
tk.Label(root, text="Typ sygnału:").grid(row=7, column=0)
signal_type.grid(row=7, column=1)

tk.Label(root, text="Parametr 1 (amplituda)").grid(row=8, column=0)
param1 = tk.Entry(root)
param1.insert(0, "1.0")
param1.grid(row=8, column=1)

tk.Label(root, text="Parametr 2 (czas/okres/częst.)").grid(row=9, column=0)
param2 = tk.Entry(root)
param2.insert(0, "1.0")
param2.grid(row=9, column=1)

tk.Button(root, text="Uruchom symulację", command=run_simulation).grid(row=10, column=0, columnspan=2, pady=10)

root.mainloop()
