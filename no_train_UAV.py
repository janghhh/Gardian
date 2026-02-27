import numpy as np
# 주의: 파일명이나 경로에 맞게 수정하세요
from onpolicy.envs.make_env_obstacle_final import AirSimMultiDroneEnv 

if __name__ == "__main__":
    # 환경 설정
    env = AirSimMultiDroneEnv(
        ip_address="127.0.0.1",
        follower_names=("Follower0", "Follower1", "Follower2"),
        # dynamic_name=("DynamicObstacle",),  <-- (참고) 이전 에러 방지용 리스트 처리
        leader_velocity=0.0,
        do_visualize=True
    )

    # 환경 초기화
    obs = env.reset()
    print("초기 obs shape:", np.array(obs).shape)
    
    # 디버깅: 바뀐 액션 공간 확인
    print(f"변경된 액션 공간 타입: {type(env.action_spaces[env.agents[0]])}")
    # 출력 결과가 <class 'gym.spaces.box.Box'> 여야 정상입니다.

    # 1000 스텝 진행
    for step in range(1000):
        
        actions = [env.action_spaces[agent].sample() for agent in env.agents]

        obs, rewards, dones, infos = env.step(actions)

        if any(dones):
            print(f"에피소드 종료 at step {step}")
            print(f"최종 보상 (각 에이전트): {rewards}")
            print(f"종료 정보 (Infos): {infos}")
            
            # 에피소드가 끝나면 다시 reset을 해줘야 계속 돌 수 있습니다. (테스트용)
            obs = env.reset()
            # break  <-- 계속 돌리고 싶으면 break를 주석 처리하세요.