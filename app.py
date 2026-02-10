import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ============================================================
# Physics (same model as your original)
# Units:
#  - distance: AU
#  - time: years
#  - mass: solar masses
#  => G_eff = 4*pi^2
# ============================================================

G = 4 * np.pi**2
DAY_IN_YEARS = 1.0 / 365.25

R_EARTH_KM = 6371.0

# Radii (km)
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

# Masses (solar masses)
M_MERKUR  = 1.651e-7
M_VENUS   = 2.447e-6
M_ERDE    = 3.003e-6
M_MARS    = 3.227e-7
M_JUPITER = 9.545e-4
M_SATURN  = 2.858e-4
M_URANUS  = 4.366e-5
M_NEPTUN  = 5.151e-5
M_MOND    = M_ERDE * 0.0123


class Body:
    def __init__(self, name, mass, a=None, e=0.0, color="white", radius_km=0.0):
        self.name = name
        self.mass = float(mass)
        self.a = a
        self.e = float(e)
        self.color = color
        self.radius_km = float(radius_km)

        # position, velocity, acceleration
        if a is None:
            self.x = self.y = self.z = 0.0
            self.vx = self.vy = self.vz = 0.0
        else:
            r_peri = a * (1 - e)
            self.x, self.y, self.z = r_peri, 0.0, 0.0
            v_peri = np.sqrt(G * (1 + e) / (a * (1 - e)))
            self.vx, self.vy, self.vz = 0.0, v_peri, 0.0

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
    ax_old = np.array([b.ax for b in bodies], dtype=float)
    ay_old = np.array([b.ay for b in bodies], dtype=float)
    az_old = np.array([b.az for b in bodies], dtype=float)

    # position update
    for k, b in enumerate(bodies):
        b.x += b.vx * dt + 0.5 * ax_old[k] * dt * dt
        b.y += b.vy * dt + 0.5 * ay_old[k] * dt * dt
        b.z += b.vz * dt + 0.5 * az_old[k] * dt * dt

    # new accelerations
    compute_accelerations(bodies)

    # velocity update
    for k, b in enumerate(bodies):
        b.vx += 0.5 * (ax_old[k] + b.ax) * dt
        b.vy += 0.5 * (ay_old[k] + b.ay) * dt
        b.vz += 0.5 * (az_old[k] + b.az) * dt


def marker_size_from_radius_km(name: str, radius_km: float) -> float:
    """Plotly marker size (pixels-ish) that mimics your compressed scaling."""
    if radius_km <= 0:
        return 6.0
    rel = radius_km / R_EARTH_KM
    size = 10.0 * (rel ** 0.3)  # stronger than matplotlib so it's visible in web
    if name == "Sonne":
        size *= 1.3
    return float(np.clip(size, 5.0, 22.0))


def build_bodies():
    bodies = []
    # Sun
    bodies.append(Body("Sonne", 1.0, a=None, color="yellow", radius_km=R_SUN))

    # Planets
    bodies.append(Body("Merkur",  M_MERKUR,  a=0.387,  e=0.2056, color="gray",        radius_km=R_MERKUR))
    bodies.append(Body("Venus",   M_VENUS,   a=0.723,  e=0.0068, color="orange",      radius_km=R_VENUS))
    bodies.append(Body("Erde",    M_ERDE,    a=1.000,  e=0.0167, color="deepskyblue", radius_km=R_ERDE))
    bodies.append(Body("Mars",    M_MARS,    a=1.524,  e=0.0934, color="red",         radius_km=R_MARS))
    bodies.append(Body("Jupiter", M_JUPITER, a=5.203,  e=0.0489, color="sandybrown",  radius_km=R_JUPITER))
    bodies.append(Body("Saturn",  M_SATURN,  a=9.537,  e=0.0542, color="gold",        radius_km=R_SATURN))
    bodies.append(Body("Uranus",  M_URANUS,  a=19.191, e=0.0472, color="cyan",        radius_km=R_URANUS))
    bodies.append(Body("Neptun",  M_NEPTUN,  a=30.07,  e=0.0086, color="blueviolet",  radius_km=R_NEPTUN))

    # Moon around Earth (same as your original)
    earth = next(b for b in bodies if b.name == "Erde")
    moon = Body("Mond", M_MOND, a=None, color="lightgray", radius_km=R_MOND)

    r_moon = 0.00257  # AU
    moon.x = earth.x + r_moon
    moon.y = earth.y
    moon.z = 0.0

    GM_earth = G * earth.mass
    v_rel = np.sqrt(GM_earth / r_moon)

    moon.vx = earth.vx
    moon.vy = earth.vy + v_rel
    moon.vz = earth.vz

    bodies.append(moon)

    compute_accelerations(bodies)
    return bodies


def make_star_field(seed=7, n=800, rmin=60.0, rmax=120.0):
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0, 2*np.pi, n)
    costheta = rng.uniform(-1, 1, n)
    u = rng.uniform(rmin**3, rmax**3, n)
    r = u ** (1/3)
    theta = np.arccos(costheta)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def orbit_line(a, e, steps=420):
    t = np.linspace(0, 2*np.pi, steps)
    x = a * np.cos(t) - a * e
    y = a * np.sin(t) * np.sqrt(1 - e**2)
    z = np.zeros_like(x)
    return x, y, z


@st.cache_data(show_spinner=False)
def simulate(days_per_step: float, n_frames: int, substeps: int):
    """
    Returns:
      names: list[str]
      colors: list[str]
      sizes: list[float]
      traj: np.ndarray shape (n_frames, n_bodies, 3)
      times_years: np.ndarray shape (n_frames,)
      stars: (sx, sy, sz)
      orbit_lines: dict[name] -> (x,y,z)
    """
    bodies = build_bodies()
    names = [b.name for b in bodies]
    colors = [b.color for b in bodies]
    sizes = [marker_size_from_radius_km(b.name, b.radius_km) for b in bodies]

    # Stars + orbit lines (static geometry)
    stars = make_star_field(seed=7, n=900)
    orbit_lines = {}
    for b in bodies:
        if b.a is not None:
            orbit_lines[b.name] = orbit_line(b.a, b.e)

    dt = (days_per_step * DAY_IN_YEARS) / max(substeps, 1)

    traj = np.zeros((n_frames, len(bodies), 3), dtype=float)
    times = np.zeros((n_frames,), dtype=float)
    sim_time = 0.0

    for f in range(n_frames):
        # advance physics
        for _ in range(substeps):
            velocity_verlet_step(bodies, dt)
            sim_time += dt

        times[f] = sim_time
        for i, b in enumerate(bodies):
            traj[f, i, :] = (b.x, b.y, b.z)

    return names, colors, sizes, traj, times, stars, orbit_lines


def build_plotly_figure(names, colors, sizes, traj, times, stars, orbit_lines, follow_name: str, limit=35.0):
    sx, sy, sz = stars

    # Follow implementation: shift everything so follow body sits at origin each frame
    if follow_name and follow_name != "Keiner" and follow_name in names:
        idx = names.index(follow_name)
        shift = traj[:, idx:idx+1, :]  # (frames,1,3)
        traj2 = traj - shift
        # Also shift stars very slightly to keep background consistent
        sx2 = sx - traj[0, idx, 0]
        sy2 = sy - traj[0, idx, 1]
        sz2 = sz - traj[0, idx, 2]
    else:
        traj2 = traj
        sx2, sy2, sz2 = sx, sy, sz

    # Base traces: stars + orbit lines + bodies markers
    data = []

    # stars
    data.append(go.Scatter3d(
        x=sx2, y=sy2, z=sz2,
        mode="markers",
        marker=dict(size=1.5, opacity=0.55),
        name="Stars",
        hoverinfo="skip"
    ))

    # orbit lines (dashed look via low opacity + thin lines)
    for name, (ox, oy, oz) in orbit_lines.items():
        # shift orbits if follow
        if follow_name and follow_name != "Keiner" and follow_name in names:
            idx = names.index(follow_name)
            ox2 = ox - traj[0, idx, 0]
            oy2 = oy - traj[0, idx, 1]
            oz2 = oz - traj[0, idx, 2]
        else:
            ox2, oy2, oz2 = ox, oy, oz

        data.append(go.Scatter3d(
            x=ox2, y=oy2, z=oz2,
            mode="lines",
            line=dict(width=2),
            opacity=0.25,
            name=f"Orbit {name}",
            hoverinfo="skip",
            showlegend=False
        ))

    # bodies (initial frame)
    x0 = traj2[0, :, 0]
    y0 = traj2[0, :, 1]
    z0 = traj2[0, :, 2]
    data.append(go.Scatter3d(
        x=x0, y=y0, z=z0,
        mode="markers+text",
        text=[n if n in ("Sonne", "Erde", "Mond") else "" for n in names],
        textposition="top center",
        marker=dict(size=sizes, color=colors, opacity=0.98),
        name="Bodies",
        hovertemplate="<b>%{text}</b><br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>"
    ))

    # Frames: update body positions only (fast)
    frames = []
    for i in range(traj2.shape[0]):
        frames.append(go.Frame(
            data=[
                go.Scatter3d(x=traj2[i, :, 0], y=traj2[i, :, 1], z=traj2[i, :, 2])
            ],
            name=str(i),
            layout=go.Layout(
                title=dict(
                    text=f"t = {times[i]:.2f} Jahre ({times[i]*365.25:.0f} Tage)",
                    x=0.02
                )
            )
        ))

    fig = go.Figure(data=data, frames=frames)

    # Slider + play controls
    fig.update_layout(
        height=720,
        margin=dict(l=0, r=0, t=45, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            bgcolor="black",
            xaxis=dict(range=[-limit, limit], showgrid=False, zeroline=False, color="white"),
            yaxis=dict(range=[-limit, limit], showgrid=False, zeroline=False, color="white"),
            zaxis=dict(range=[-limit, limit], showgrid=False, zeroline=False, color="white"),
            aspectmode="cube",
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.02, y=0.02,
                xanchor="left", yanchor="bottom",
                buttons=[
                    dict(label="Play", method="animate",
                         args=[None, {"frame": {"duration": 25, "redraw": False},
                                      "transition": {"duration": 0},
                                      "fromcurrent": True, "mode": "immediate"}]),
                    dict(label="Pause", method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate"}])
                ],
            )
        ],
        sliders=[
            dict(
                x=0.18, y=0.03, len=0.78,
                currentvalue=dict(prefix="Frame: "),
                steps=[dict(method="animate", args=[[str(k)], {"mode": "immediate",
                                                             "frame": {"duration": 0, "redraw": False},
                                                             "transition": {"duration": 0}}],
                            label=str(k)) for k in range(traj2.shape[0])]
            )
        ],
        title=dict(text=f"t = {times[0]:.2f} Jahre ({times[0]*365.25:.0f} Tage)", x=0.02)
    )

    return fig


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="Universe Simulator · Marcos Caprile-Santos", layout="wide")

st.title("Universe Simulator")
st.caption("Interaktives 3D-Showcase (Browser-Drag rotate, Wheel zoom) · N-Body Physics (Velocity-Verlet)")

with st.sidebar:
    st.header("Controls")
    days_per_step = st.slider("Tage pro Schritt", 0.1, 50.0, 1.0, 0.1)
    follow = st.selectbox("Follow (zentrieren)", ["Keiner", "Merkur", "Venus", "Erde", "Mond", "Mars", "Jupiter", "Saturn", "Uranus", "Neptun"])
    n_frames = st.slider("Frames (mehr = flüssiger, aber langsamer)", 120, 700, 320, 20)
    substeps = st.slider("Substeps pro Frame (mehr = stabiler)", 1, 20, 6, 1)
    st.markdown("---")
    st.write("Tip: Wenn es auf Streamlit Cloud ruckelt, setz Frames runter oder Substeps etwas runter.")

names, colors, sizes, traj, times, stars, orbit_lines = simulate(days_per_step, n_frames, substeps)

fig = build_plotly_figure(
    names=names,
    colors=colors,
    sizes=sizes,
    traj=traj,
    times=times,
    stars=stars,
    orbit_lines=orbit_lines,
    follow_name=follow,
    limit=35.0
)

st.plotly_chart(fig, use_container_width=True)

st.markdown(
    """
**Bedienung**
- **Drag** = Rotate  
- **Mausrad** = Zoom  
- **Play/Pause** unten links  
- **Follow** zentriert das Koordinatensystem auf einen Körper (wie „Kamera folgt“).
"""
)
