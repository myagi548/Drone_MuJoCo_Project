import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("drone_model.xml")
data = mujoco.MjData(model)

mass = model.body_mass[1]
gravity = 9.81

total_hover_thrust = mass * gravity
rotor_thrust = total_hover_thrust / 4.0

print("Drone mass =", mass, "kg")
print("Gravity =", gravity, "m/s^2")
print("Total hover thrust =", total_hover_thrust, "N")
print("Thrust per rotor =", rotor_thrust, "N")

data.ctrl[0] = rotor_thrust
data.ctrl[1] = rotor_thrust
data.ctrl[2] = rotor_thrust
data.ctrl[3] = rotor_thrust

with mujoco.viewer.launch_passive(model, data) as viewer:

    next_print = 0.5

    while viewer.is_running() and data.time < 5.0:

        mujoco.mj_step(model, data)

        if data.time >= next_print:
            print(
                f"Time = {data.time:.1f} s | "
                f"Drone Z = {data.qpos[2]:.3f} m"
            )
            next_print += 0.5

        viewer.sync()