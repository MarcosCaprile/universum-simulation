# app.py
# Streamlit Universe Simulator (Interactive 3D) – Plotly + Play/Pause + Follow (camera)
# - Drag to rotate, scroll to zoom, right-drag to pan (Plotly default)
# - Planets + Moon selectable for follow
# - Stars background
#
# requirements.txt:
# streamlit
# numpy
# plotly

import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go


# =========================
# Physics (your real N-body core)
# Units:
#  - distance: AU
#  - time: years
#  - mass: solar masses
#  -> G_eff = 4*pi^2
# =========================
G = 4 * np.pi**2
DAY_IN_YEARS = 1.0 / 365.25
AU_IN_KM = 1.496e8
R_EARTH_KM = 6371.0


class Body:
    def __init__(self, name, mass, a=None, e=0.0, inc_deg=0.0, color="#ffffff", radius_km=0.0):
        self.name = name
        self.mass = mass
        self.color = color
        self.radius_km = radius_km
        self.a = a
        self.e = e
        self.inc_deg = inc_deg

        # state
        if a is None:
            self.x = self.y = self.z = 0.0
            self.vx = self.vy = self.vz = 0.0
        else:
            # start at perihelion (z=0)
            r_peri = a * (1 - e)
            self.x = r_peri
            self.y = 0.0
            self.z = 0.0

            # perihelion speed around Sun (approx)
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

            r2 = dx * dx + dy * dy + dz * dz
            r = np.sqrt(r2) + 1e-12

            factor = G * bj.mass / (r**3)
            bi.ax += factor * dx
            bi.ay += factor * dy
            bi.az += factor * dz


def velocity_verlet_step(bodies, dt):
    ax_old = [b.ax for b in bodies]
    ay_old = [b.ay for b in bodies]
    az_old = [b.az for b in bodies]

    # position
    for i, b in enumerate(bodies):
        b.x += b.vx * dt + 0.5 * ax_old[i] * dt * dt
        b.y += b.vy * dt + 0.5 * ay_old[i] * dt * dt
        b.z += b.vz * dt + 0.5 * az_old[i] * dt * dt

    # acceleration
    compute_accelerations(bodies)

    # velocity
    for i, b in enumerate(bodies):
        b.vx += 0.5 * (ax_old[i] + b.ax) * dt
        b.vy += 0.5 * (ay_old[i] + b.ay) * dt
        b.vz += 0.5 * (az_old[i] + b.az) * dt


def marker_size_from_radius(body: Body) -> float:
    """Marker size for Plotly: compressed size range so everything stays visible."""
    if body.radius_km <= 0:
        return 4.0
    rel = body.radius_km / R_EARTH_KM
    size = 9.0 * (rel ** 0.3)  # compressed scale
    if body.name == "Sonne":
        size *= 1.45
    return float(np.clip(size, 4.0, 26.0))


def build_bodies():
    bodies = []

    # radii (km)
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

    # masses (solar masses)
    M_MERKUR  = 1.651e-7
    M_VENUS   = 2.447e-6
    M_ERDE    = 3.003e-6
    M_MARS    = 3.227e-7
    M_JUPITER = 9.545e-4
    M_SATURN  = 2.858e-4
    M_URANUS  = 4.366e-5
    M_NEPTUN  = 5.151e-5
    M_MOND    = M_ERDE * 0.0123

    # Sun
    sun = Body("Sonne", mass=1.0, a=None, color="#ffd24a", radius_km=R_SUN)
    bodies.append(sun)

    # Planets
    bodies.append(Body("Merkur",  M_MERKUR,  a=0.387,  e=0.2056, inc_deg=7.0,  color="#9aa0a6", radius_km=R_MERKUR))
    bodies.append(Body("Venus",   M_VENUS,   a=0.723,  e=0.0068, inc_deg=3.4,  color="#ff9f4a", radius_km=R_VENUS))
    bodies.append(Body("Erde",    M_ERDE,    a=1.000,  e=0.0167, inc_deg=0.0,  color="#3aa8ff", radius_km=R_ERDE))
    bodies.append(Body("Mars",    M_MARS,    a=1.524,  e=0.0934, inc_deg=1.85, color="#ff3a2f", radius_km=R_MARS))
    bodies.append(Body("Jupiter", M_JUPITER, a=5.203,  e=0.0489, inc_deg=1.3,  color="#d9a06a", radius_km=R_JUPITER))
    bodies.append(Body("Saturn",  M_SATURN,  a=9.537,  e=0.0542, inc_deg=2.49, color="#d7b05a", radius_km=R_SATURN))
    bodies.append(Body("Uranus",  M_URANUS,  a=19.191, e=0.0472, inc_deg=0.77, color="#6fe7ff", radius_km=R_URANUS))
    bodies.append(Body("Neptun",  M_NEPTUN,  a=30.07,  e=0.0086, inc_deg=1.77, color="#7a4dff", radius_km=R_NEPTUN))

    # Moon around Earth (initial)
    earth = next(b for b in bodies if b.name == "Erde")
    moon = Body("Mond", M_MOND, a=None, color="#d0d0d0", radius_km=R_MOND)

    r_moon_au = 384400.0 / AU_IN_KM  # ~0.00257 AU
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


# =========================
# Stars background (static)
# =========================
def make_stars(seed=42, n=1200, rmin=60, rmax=120):
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0, 2 * np.pi, n)
    costheta = rng.uniform(-1, 1, n)
    u = rng.uniform(rmin**3, rmax**3, n)
    r = u ** (1 / 3)

    theta = np.arccos(costheta)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


# =========================
# Plotly Figure
# =========================
def render_plotly(bodies, stars_xyz, view_limit=35, follow_name="Keiner", prev_camera=None):
    # body arrays
    xs = [b.x for b in bodies]
    ys = [b.y for b in bodies]
    zs = [b.z for b in bodies]
    cols = [b.color for b in bodies]
    names = [b.name for b in bodies]
    sizes = [marker_size_from_radius(b) for b in bodies]

    # stars
    sx, sy, sz = stars_xyz

    # Find follow target
    fx = fy = fz = 0.0
    if follow_name and follow_name != "Keiner":
        for b in bodies:
            if b.name == follow_name:
                fx, fy, fz = b.x, b.y, b.z
                break

    # Build figure
    fig = go.Figure()

    # Stars first (background)
    fig.add_trace(
        go.Scatter3d(
            x=sx, y=sy, z=sz,
            mode="markers",
            marker=dict(size=1.4, color="white", opacity=0.55),
            hoverinfo="skip",
            name="Stars",
        )
    )

    # Bodies
    fig.add_trace(
        go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="markers+text",
            marker=dict(size=sizes, color=cols, opacity=1.0),
            text=names,
            textposition="top center",
            textfont=dict(color="rgba(255,255,255,0.85)", size=10),
            hovertemplate="%{text}<extra></extra>",
            name="Bodies",
        )
    )

    # Layout / scene
    scene = dict(
        xaxis=dict(
            range=[fx - view_limit, fx + view_limit],
            backgroundcolor="black",
            gridcolor="rgba(255,255,255,0.08)",
            zerolinecolor="rgba(255,255,255,0.14)",
            color="white",
        ),
        yaxis=dict(
            range=[fy - view_limit, fy + view_limit],
            backgroundcolor="black",
            gridcolor="rgba(255,255,255,0.08)",
            zerolinecolor="rgba(255,255,255,0.14)",
            color="white",
        ),
        zaxis=dict(
            range=[fz - view_limit, fz + view_limit],
            backgroundcolor="black",
            gridcolor="rgba(255,255,255,0.08)",
            zerolinecolor="rgba(255,255,255,0.14)",
            color="white",
        ),
        aspectmode="cube",
    )

    fig.update_layout(
        scene=scene,
        margin=dict(l=0, r=0, t=0, b=0),
        height=720,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="black",
        showlegend=False,
        uirevision="keep"  # keep user zoom/rotate across reruns
    )

    # Preserve camera if we have one
    if prev_camera is not None:
        fig.update_layout(scene_camera=prev_camera)

    return fig


# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="Universe Simulator · Marcos Caprile-Santos", layout="wide")

st.title("Universe Simulator")
st.caption("Interaktives 3D-Showcase (echte N-Body-Physik) – Drag/Zoom/Pan + Follow Planet/Mond + Sterne.")

# ---------- Session state ----------
if "bodies" not in st.session_state:
    st.session_state.bodies = build_bodies()

if "sim_time" not in st.session_state:
    st.session_state.sim_time = 0.0

if "running" not in st.session_state:
    st.session_state.running = False

if "stars" not in st.session_state:
    st.session_state.stars = make_stars(seed=7, n=1400)

if "camera" not in st.session_state:
    st.session_state.camera = None  # Plotly camera snapshot


# ---------- UI ----------
left, right = st.columns([1, 2], gap="large")

with left:
    st.subheader("Controls")

    days_per_step = st.slider("Tage pro Schritt", 0.1, 50.0, 1.0, 0.1)
    steps_per_frame = st.slider("Steps pro Frame", 1, 60, 12, 1)
    view_limit = st.slider("View / Zoom Limit (AU)", 5, 80, 35, 1)

    follow_options = ["Keiner"] + [b.name for b in st.session_state.bodies if b.name != "Sonne"]
    follow_name = st.selectbox("Follow", follow_options, index=0)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Reset Simulation"):
            st.session_state.bodies = build_bodies()
            st.session_state.sim_time = 0.0
            st.session_state.camera = None
    with c2:
        if st.button("Reset View"):
            st.session_state.camera = None

    st.divider()

    colPlay, colFps = st.columns([1, 1])
    with colPlay:
        if st.button("▶ Play" if not st.session_state.running else "⏸ Pause"):
            st.session_state.running = not st.session_state.running

    with colFps:
        fps = st.slider("FPS", 1, 30, 18, 1)

    st.write("")
    st.metric("Sim-Zeit (Jahre)", f"{st.session_state.sim_time:.3f}")
    st.caption("Tipp: Wenn es ruckelt, senke FPS oder Steps pro Frame.")

with right:
    st.subheader("3D View")
    time_days = st.session_state.sim_time * 365.25
    st.write(f"**t = {st.session_state.sim_time:.2f} Jahre**  ({time_days:.0f} Tage)")

    # ---- Advance physics for this render ----
    dt = days_per_step * DAY_IN_YEARS
    for _ in range(steps_per_frame):
        velocity_verlet_step(st.session_state.bodies, dt)
        st.session_state.sim_time += dt

    # ---- Build plotly figure ----
    fig = render_plotly(
        st.session_state.bodies,
        st.session_state.stars,
        view_limit=view_limit,
        follow_name=follow_name,
        prev_camera=st.session_state.camera,
    )

    # Capture camera changes (best-effort):
    # Streamlit can't directly read client-side camera state without extra components.
    # We still keep uirevision="keep" so user interactions persist.
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

# ---------- Auto update ----------
if st.session_state.running:
    time.sleep(1.0 / float(fps))
    st.rerun()
