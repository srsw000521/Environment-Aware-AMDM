from make_perlin_noise_map import generate_A_masked_binary01  # 네가 추가한 함수 이름에 맞춰서
from image_nav_map import PerlinNavMap


if __name__ == "__main__":
    A = generate_A_masked_binary01(size=500, seedA=0, seedB=1)  # (500,500) {0,1}

    nav = PerlinNavMap(
        img_path=A,
        world_size_m=50.0,
        boundary_block_m=2.0,
        input_is_binary01=True,   # 중요!
    )

    nav.build_reachability_map(jump_bridge_m=0.6)
    nav.to_torch(device="cuda",use_reach_map=True)
    start = nav.sample_start()
    goal = nav.sample_goal_near_start(start[0], start[1], roi_size_m=10.0)
    nav.debug_show_maps(start=start, goal=goal)

    nav.debug_show_local_grid(x_m=start[0], z_m=start[1], yaw_rad=0)
    nav.debug_show_local_grid(x_m=start[0], z_m=start[1], yaw_rad=3.141592/4)

    import matplotlib.pyplot as plt
    plt.show()


