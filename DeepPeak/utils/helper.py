def get_throughput_label(particle_flow: float):
    throughput_label = (
        f"{particle_flow / 1000:.1f}k"
        if particle_flow >= 1000 and particle_flow < 1_000_000
        else (
            f"{particle_flow / 1_000_000:.1f}M"
            if particle_flow >= 1_000_000
            else f"{particle_flow:.0f}"
        )
    )

    return throughput_label
