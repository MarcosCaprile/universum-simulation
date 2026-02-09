# app.py
import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")  # wichtig: Headless rendering
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ====== DEIN PHYSIK-TEIL (gekürzt/angepasst) ======
G = 4 * np.pi**2
DAY_IN_YEARS = 1.0 / 365.25
AU_IN_KM = 1.496e8
R_EARTH_KM = 6371.0


class Body:
    def __init__(self, name, mass, a=None, e=0.0, inc_deg=0.0, color="white", radius_km=0.0):
        self.name = name
        self.mass = mass
        self.color = color
        self.radius_km = radius_km
        self.a = a
        self.e = e
        self.inc_deg = inc_deg

        if a is None:
            self.x = self.y = self.z = 0.0
            self.vx = self.vy = self.vz = 0.0
        else:
            r_peri = a * (1 - e)
            self.x = r_peri
            self.y = 0.0
            self.z = 0.0
            v_peri = np.sqrt(G * (1 + e) / (a * (1 - e)))
            self.vx = 0.0
            self.vy = v_peri
            self.vz = 0.0

        self.ax = self.ay = self.az = 0.0


def compute_accelerations(bodies):
    n = len(bodies)
    for b in bodies:
        b.ax = b.ay = b.az = 0.0

    for i in range(n):
        bi = bodies[i]
        for j in range(n):
            if i == j:
                continue
            bj = bodies[j]
            dx = bj.x - bi.x
            dy = bj.y - bi.y
            dz = bj.z - bi.z
            r2 = dx*dx + dy*dy + dz*dz
            r = np.sqrt(r2) + 1e-12
            factor = G * bj.mass / (r**3)
            bi.ax += factor * dx
            bi.ay += factor * dy
            bi.az += factor * dz


def velocity_verlet_step(bodies, dt):
    ax_old = [b.ax for b in bodies]
    ay_old = [b.ay for b in bodies]
    az_old = [b.az for b in bodies]

    for i, b in enumerate(bodies):
        b.x += b.vx * dt + 0.5 * ax_old[i] * dt * dt
        b.y += b.vy * dt + 0.5 * ay_old[i] * dt * dt
        b.z += b.vz * dt + 0.5 * az_old[i] * dt * dt

    compute_accelerations(bodies)

    for i, b in enumerate(bodies):
        b.vx += 0.5 * (ax_old[i] + b.ax) * dt
        b.vy += 0.5 * (ay_old[i] + b.ay) * dt
        b.vz += 0.5 * (az_old[i] + b.az) * dt


def marker_size_from_radius(body):
    if body.radius_km <= 0:
        return 6.0
    rel = body.radius_km / R_EARTH_KM
    size = 4.0 * (rel ** 0.3)
    if body.name == "Sonne":
        size *= 1.5
    return float(np.clip(size, 3.0, 20.0))


def build_bodies():
    bodies = []

    # Radien (km)
    R_SUN     = 695700.0
    R_MERKUR  = 2439.7
    R_VENUS   = 6051.8
    R_ERDE    = 6371.0
    R_MARS    = 3389.5
    R_JUPITER = 69911.0
    R_SATURN  = 58232.0
    R_URANUS  = 25362.0
    R_NEPTUN  = 24622.0
    R_MOND    = 1737.4

    sun = Body("Sonne", 1.0, a=None, color="yellow", radius_km=R_SUN)
    bodies.append(sun)

    # Massen (Sonnenmassen)
    M_MERKUR  = 1.651e-7
    M_VENUS   = 2.447e-6
    M_ERDE    = 3.003e-6
    M_MARS    = 3.227e-7
    M_JUPITER = 9.545e-4
    M_SATURN  = 2.858e-4
    M_URANUS  = 4.366e-5
    M_NEPTUN  = 5.151e-5
    M_MOND    = M_ERDE * 0.0123

    bodies.append(Body("Merkur",  M_MERKUR,  a=0.387,  e=0.2056, color="gray",        radius_km=R_MERKUR))
    bodies.append(Body("Venus",   M_VENUS,   a=0.723,  e=0.0068, color="orange",      radius_km=R_VENUS))
    bodies.append(Body("Erde",    M_ERDE,    a=1.000,  e=0.0167, color="deepskyblue", radius_km=R_ERDE))
    bodies.append(Body("Mars",    M_MARS,    a=1.524,  e=0.0934, color="red",         radius_km=R_MARS))
    bodies.append(Body("Jupiter", M_JUPITER, a=5.203,  e=0.0489, color="sandybrown",  radius_km=R_JUPITER))
    bodies.append(Body("Saturn",  M_SATURN,  a=9.537,  e=0.0542, color="gold",        radius_km=R_SATURN))
    bodies.append(Body("Uranus",  M_URANUS,  a=19.191, e=0.0472, color="cyan",        radius_km=R_URANUS))
    bodies.append(Body("Neptun",  M_NEPTUN,  a=30.07,  e=0.0086, color="blueviolet",  radius_km=R_NEPTUN))

    earth = next(b for b in bodies if b.name == "Erde")

    moon = Body("Mond", M_MOND, a=None, color="lightgray", radius_km=R_MOND)
    r_moon_au = 384400.0 / AU_IN_KM
    moon.x = earth.x + r_moon_au
    moon.y = earth.y
    moon.z = earth.z

    GM_earth = G * earth.mass
    v_moon_rel = np.sqrt(GM_earth / r_moon_au)

    moon.vx = earth.vx
    moon.vy = earth.vy + v_moon_rel
    moon.vz = earth.vz

    bodies.append(moon)

    compute_accelerations(bodies)
    return bodies


def render_figure(bodies, limit=35, elev=25, azim=45):
    fig = plt.figure(figsize=(7.5, 7.5))
    ax = fig.add_subplot(111, projection="3d")

    ax.set_facecolor("black")
    ax.grid(False)

    # panes transparent
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("white")

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_box_aspect((1, 1, 1))
    ax.tick_params(colors="white")

    ax.set_xlabel("x (AU)", color="white")
    ax.set_ylabel("y (AU)", color="white")
    ax.set_zlabel("z (AU)", color="white")

    ax.view_init(elev=elev, azim=azim)

    # stars (schnell)
    n_stars = 500
    star_radius_min = 60
    star_radius_max = 120
    phi = np.random.uniform(0, 2*np.pi, n_stars)
    costheta = np.random.uniform(-1, 1, n_stars)
    u = np.random.uniform(star_radius_min**3, star_radius_max**3, n_stars)
    r = u**(1/3)
    theta = np.arccos(costheta)
    sx = r * np.sin(theta) * np.cos(phi)
    sy = r * np.sin(theta) * np.sin(phi)
    sz = r * np.cos(theta)
    ax.scatter(sx, sy, sz, s=1, color="white", alpha=0.5, zorder=0)

    # bodies
    for b in bodies:
        ms = marker_size_from_radius(b)
        ax.plot([b.x], [b.y], [b.z], "o", color=b.color, markersize=ms, zorder=5)

    plt.tight_layout()
    return fig


# ====== STREAMLIT UI ======
st.set_page_config(page_title="Universe Simulator", layout="wide")

st.title("Universe Simulator")
st.caption("Interaktives Showcase deines echten Python-Simulators – als Streamlit App.")

colA, colB = st.columns([1, 2], gap="large")

with colA:
    st.subheader("Controls")
    days_per_step = st.slider("Tage pro Schritt", 0.1, 50.0, 1.0, 0.1)
    steps = st.slider("Steps pro Update", 1, 50, 10, 1)
    limit = st.slider("Zoom / View Limit (AU)", 5, 60, 35, 1)
    elev = st.slider("Kamera Elevation", 0, 90, 25, 1)
    azim = st.slider("Kamera Azimuth", 0, 360, 45, 1)

    if "bodies" not in st.session_state:
        st.session_state.bodies = build_bodies()
        st.session_state.sim_time = 0.0

    if st.button("Reset Simulation"):
        st.session_state.bodies = build_bodies()
        st.session_state.sim_time = 0.0

    st.write("")
    st.metric("Sim-Zeit (Jahre)", f"{st.session_state.sim_time:.3f}")

with colB:
    st.subheader("View")

    dt = days_per_step * DAY_IN_YEARS
    for _ in range(steps):
        velocity_verlet_step(st.session_state.bodies, dt)
        st.session_state.sim_time += dt

    fig = render_figure(st.session_state.bodies, limit=limit, elev=elev, azim=azim)
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

st.info("Hinweis: Streamlit ist kein 60fps-Animator. Dafür ist es perfekt als interaktives Showcase mit echten Physik-Steps.")

