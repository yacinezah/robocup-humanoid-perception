# Authorship and component licences

This portfolio was developed by Yacine Zahouani during a research programme
with ITA-ITAndroids, supervised by Marcos Maximo. The robot platform, team
software, established model architectures and source datasets predate this work.
They are not claimed as original inventions.

| Component | Attribution and conditions |
|---|---|
| Original research utilities, field-context module and analysis code | Yacine Zahouani, MIT, except inherited portions identified below |
| `src/soccer_segmentation/` | Inherited research repository, MIT, copyright Otavio H. Ribas (2022), with subsequent project changes. Preserve [its licence](LICENSES/segmentation-upstream-MIT.txt) |
| Ultralytics YOLO models and framework | Ultralytics and contributors. External dependency, not vendored. Preserve upstream terms; importing these wrappers does not grant a commercial or permissive licence to the full dependency/model stack |
| `segmentation_models_pytorch`, torchvision and timm | External architecture dependencies, not vendored. Consult the pinned environment and upstream notices before redistributing a combined product |
| ITAndroids C++ patch context | Existing team code remains under its original ownership/terms. Public release permission covers this review material; the portfolio MIT licence does not relicense inherited C++ lines |
| Gazebo/ROS robot assets | External team/ROS/Gazebo dependencies. Robot meshes, complete worlds and full upstream repositories are not mirrored here |
| Original reports, charts and aggregate tables | Yacine Zahouani, CC BY 4.0 |
| Training images, masks and captures | Not included. Publication of code does not grant redistribution rights to source datasets |

Model bundles have a separate rights/status entry in the artifact index. A
checkpoint's availability is not evidence of a new licence. If a model's
redistribution terms cannot be established, its metadata may be published but
its binary is withheld. No production robot model is included by default.

The original mail patch preserves its historical author header and SHA-256;
that archival header is provenance, not a new assertion of sole authorship.
