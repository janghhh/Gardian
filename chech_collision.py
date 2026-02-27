import airsim
import time

# 드론 이름
agent = "Follower0"
enemy = "DynamicObstacle"
fixed_z = -5

client = airsim.MultirotorClient()
client.confirmConnection()

# API 제어
client.enableApiControl(True, agent)
client.enableApiControl(True, enemy)

client.armDisarm(True, agent)
client.armDisarm(True, enemy)

print("[INFO] Taking off...")
client.takeoffAsync(vehicle_name=agent).join()
client.takeoffAsync(vehicle_name=enemy).join()

# 고도 통일
client.moveToPositionAsync(0, 0, fixed_z, 3, vehicle_name=agent).join()
client.moveToPositionAsync(5, 0, fixed_z, 3, vehicle_name=enemy).join()

# 초기 충돌 이벤트 클리어
print("[INFO] Clearing initial collision events...")
time.sleep(0.3)
for _ in range(10):
    client.simGetCollisionInfo(agent)
    client.simGetCollisionInfo(enemy)
    time.sleep(0.05)

print("\n[INFO] Ready.")
print("[INFO] Move the drones in Unreal manually.")
print("[INFO] Waiting for a collision event ...\n")

# ---------------------------------------------------------
# 충돌 발생까지 대기하는 루프 (이동 없음)
# ---------------------------------------------------------
while True:

    colA = client.simGetCollisionInfo(agent)
    colE = client.simGetCollisionInfo(enemy)

    if colA.has_collided:
        print("🔥 [충돌 EVENT] Agent 충돌 발생!")
        print(f"  object_name = {colA.object_name}")
        print(f"  object_id   = {colA.object_id}")
        print(f"  impact_pos  = ({colA.position.x_val:.2f}, {colA.position.y_val:.2f}, {colA.position.z_val:.2f})")
        break

    if colE.has_collided:
        print("🔥 [충돌 EVENT] Enemy 충돌 발생!")
        print(f"  object_name = {colE.object_name}")
        print(f"  object_id   = {colE.object_id}")
        print(f"  impact_pos  = ({colE.position.x_val:.2f}, {colE.position.y_val:.2f}, {colE.position.z_val:.2f})")
        break

    time.sleep(0.1)

# 충돌 후 정지
client.hoverAsync(agent)
client.hoverAsync(enemy)

print("\n[INFO] Test finished.")
