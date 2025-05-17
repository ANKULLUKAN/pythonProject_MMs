import numpy as np
import matplotlib.pyplot as plt


def u_step(t, amplitude, duration):
    return amplitude if t < duration else 0

def u_triangle(t, amplitude, period):
    return amplitude * (2/period) * (period/2 - abs((t % period) - period/2))

def u_sine(t, amplitude, frequency):
    return amplitude * np.sin(2 * np.pi * frequency * t)

# model
def model(x, t, u_func, M, b, k):
    x1, x2 = x
    u = u_func(t)
    dx1dt = x2
    dx2dt = (u - b * x2 - k * x1) / M
    return np.array([dx1dt, dx2dt])

# Eulera
def euler(x0, u_func, dt, T, M, b, k):
    t = np.arange(0, T, dt)
    x = np.zeros((len(t), len(x0)))
    x[0] = x0
    for i in range(1, len(t)):
        x[i] = x[i-1] + dt * model(x[i-1], t[i-1], u_func, M, b, k)
    return t, x

#Runge Kutta 4
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


def main(T, M, b, k, fi_deg, dt, u_func):
    fi_rad = fi_deg * np.pi / 180
    x0 = [-r * fi_rad, 0.0]
    print("warunki poczatkowe")
    print(x0)

    # Symulacja
    t_euler, x_euler = euler(x0, u_func, dt, T, M, b, k)
    t_rk4, x_rk4 = rk4(x0, u_func, dt, T, M, b, k)

    # Wartosc sygn wej
    u_values = np.array([u_func(ti) for ti in t_euler])

    plt.figure(figsize=(12, 8))

    # Wykres u(t)
    plt.subplot(3, 1, 1)
    plt.plot(t_euler, u_values, label='Wymuszenie u(t)', color='purple')
    plt.ylabel('u(t)')
    plt.title('Sygnał wejściowy')
    plt.grid()
    plt.legend()

    # Wykres x(t)
    plt.subplot(3, 1, 2)
    plt.plot(t_euler, x_euler[:, 0], label='Euler  x(t)')
    plt.plot(t_rk4, x_rk4[:, 0], label='RK4  x(t)', linestyle='--')
    plt.ylabel('Pozycja x(t) [m]')
    plt.title('Pozycja układu')
    plt.grid()
    plt.legend()

    # Wykres prędkości v(t)
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
    return (T, M, b, k, fi_deg, r)


# interfejs opis wer 1

print("Podaj parametry symulacji:")
M = float(input("Masa M (kg): "))
b = float(input("Tłumienie b (Ns/m): "))
k = float(input("Sprężystość k (N/m): "))
r = float(input("Promień r (m) : "))
dt = float(input("Krok czasowy dt (s) : "))
T = float(input("Czas symulacji T (s) : "))
fi_deg = float(input("Podaj początkowy kąt θ [deg] (domyślnie 0): ") or 0.0)

# sygnal wejsc
print("\nWybierz sygnał wejściowy:")
print("1 - Skok")
print("2 - Trójkąt")
print("3 - Sinus")
choice = input("Twój wybór : ")

if choice == "1":
    amplitude = float(input("Amplituda skoku : "))
    duration = float(input("Czas trwania skoku : "))
    u_func = lambda t: u_step(t, amplitude=amplitude, duration=duration)
elif choice == "2":
    amplitude = float(input("Amplituda trójkąta : "))
    period = float(input("Okres trójkąta : "))
    u_func = lambda t: u_triangle(t, amplitude=amplitude, period=period)
else:
    amplitude = float(input("Amplituda sinusa : "))
    frequency = float(input("Częstotliwość sinusa (Hz) : "))
    u_func = lambda t: u_sine(t, amplitude=amplitude, frequency=frequency)

main(T, M, b, k, fi_deg, dt, u_func)

while True:
    answer = input("Czy chcesz zmienić parametr i przeliczyć symulację? (tak lub nie): ").lower()
    if answer != 'tak':
        print("koniec")
        break

    param_to_change = input("Który parametr chcesz zmienić? (M / b/ k / r / dt / T / θ[fi]): ").lower()

    if param_to_change == 'm':
        M = float(input("Nowa wartość dla M: "))
    elif param_to_change == 'b':
        b = float(input("Nowa wartość dla b: "))
    elif param_to_change == 'k':
        k = float(input("Nowa wartość dla k: "))
    elif param_to_change == 'r':
        r = float(input("Nowa wartość dla r: "))
    elif param_to_change == 'dt':
        dt = float(input("Nowa wartość dla dt: "))
    elif param_to_change == 't':
        T = float(input("Nowa wartość dla T: "))
    elif param_to_change == 'fi':
        fi_deg = float(input("Nowa wartość dla fi: "))
    else:
        print("Nieznany parametr.")
        continue

    main(T, M, b, k, fi_deg, dt, u_func)
