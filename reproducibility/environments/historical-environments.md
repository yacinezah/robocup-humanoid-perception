# Historical environments

| Operation | Recorded environment |
|---|---|
| Selective-sharing training | Torch 2.11.0+cu130, Ultralytics 8.4.96, RTX5090 |
| Selective-sharing laptop export | Torch 2.5.1+cu121, Ultralytics 8.4.60, OpenVINO 2026.2.0 |
| Actual Chape detector timing | Intel i5-4250U, Ubuntu 20.04.6, OpenVINO 2024.2 |
| Selective-sharing laptop timing | Intel i5-12450H, OpenVINO 2026.2.0, affinity [0,2,4,6] |
| Gazebo vision integration | Gazebo 11, ROS Noetic, team base 6740b3321d6521721f4e2937d0008a5f0d1f4c02 |
| Source observability audit | Team reference fb58697e627a92f0ca64ed74738438601ee49751 |

These records are environment summaries, not complete executable lockfiles.
Publication validation uses an isolated CPU environment, recorded separately.
It does not retroactively certify any earlier numerical or timing result.
