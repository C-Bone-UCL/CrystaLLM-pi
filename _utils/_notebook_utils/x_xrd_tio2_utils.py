
TOP_K_PEAKS = 20
THETA_MIN, THETA_MAX = 0, 90
INTENSITY_MIN, INTENSITY_MAX = 0, 100

def process_xrd_to_condition_vector(file_content: str) -> str:
    """Convert a simple two-column XRD text pattern to a notebook condition vector."""
    lines = file_content.strip().split("\n")
    data_lines = lines[1:] if len(lines) > 1 else lines

    peaks = []
    for line in data_lines:
        if not line.strip():
            continue

        parts = line.strip().split()
        if len(parts) < 2:
            continue

        try:
            two_theta = float(parts[0])
            intensity = float(parts[1])
        except ValueError:
            continue

        peaks.append({"two_theta": two_theta, "intensity": intensity})

    peaks = sorted(peaks, key=lambda entry: entry["intensity"], reverse=True)[:TOP_K_PEAKS]
    thetas = [entry["two_theta"] for entry in peaks]
    intensities = [entry["intensity"] for entry in peaks]

    thetas += [-100] * (TOP_K_PEAKS - len(thetas))
    intensities += [-100] * (TOP_K_PEAKS - len(intensities))

    valid_intensities = [value for value in intensities if value != -100]
    intensity_max = max(valid_intensities) if valid_intensities else 1

    scaled_theta = [
        round((theta - THETA_MIN) / (THETA_MAX - THETA_MIN), 3) if theta != -100 else -100
        for theta in thetas
    ]
    scaled_intensity = [
        round(value / intensity_max, 3) if value != -100 else -100
        for value in intensities
    ]

    vector_str = ",".join(map(str, scaled_theta + scaled_intensity))
    print(
        "Theta scaled to [0,1] (0 to 90), Intensity scaled to [0,1] "
        "(relative to max in pattern), -100 for padding"
    )
    return vector_str

__all__ = []