# Public-edition validation

This checks the publication, not new scientific performance. Historical results
retain the protocols, parity warnings and evidence classes stated in the reports.

## Checks completed

- 25 focused CPU tests passed, including neural gradient routing and finite
  output checks, coordinate/label handling, association and analytic line tests.
- The dataset-free synthetic example passed.
- Both selected multitask checkpoints loaded strictly into the extracted model
  classes. Synthetic forwards were finite and both outputs depended on input.
- Extracted field architectures, detector matching metrics and field-context
  functions passed definition-level AST comparisons against archived source.
- Eight model bundles passed ONNX checking and OpenVINO load/synthetic checks.
  Constant detector placeholders in the binary field graphs remain explicitly
  identified; they are not counted as functional detectors.
- The C++ mail patch retains SHA-256
  `f4adc6807e7ed9254029cc6cdcd44bb853574b34f89001a61a949d4bc5e70ddd`.
- Sixteen technical-report PDFs, a four-page engineering overview and all ten
  presentation slides were rendered and visually inspected. The editable PPTX
  was also exported through LibreOffice and its ten-page PDF inspected.
- Relative documentation links, Python syntax, public study names and private
  machine paths were checked. Release ZIP CRC and every decompressed member's
  SHA-256 were verified.

## Metadata-only public derivatives

Three detector ONNX exports embedded private training-host paths in their
description metadata. Their public copies replace that description only.
The serialized model with metadata removed is byte-identical before and after;
weights, graph and numerical behavior are unchanged. The artifact index and
bundle notes preserve archival and public SHA-256 values. Corresponding
OpenVINO IR files are unchanged.

## Scope limitations

This publication did not rebuild the ROS/Gazebo stack or rerun the C++ component
test on the robot. Those results are archived engineering evidence, not new
validation claims. It did not rerun accuracy, protected evaluations, training or
hardware timing. The public Python test suite is not a substitute for upstream
integration review. Historical strict-parity failures remain failures.

The published code includes faithful extracted functions as well as newly
written portable CLI wrappers. It is not claimed to be a byte-identical copy of
every historical workstation runner. See source-provenance.json and
extraction-validation.json for the narrower equivalence checks actually made.

## Local publication environment

Windows; Python 3.12; CPU Torch 2.5.1; Ultralytics 8.4.60; OpenVINO 2026.2;
ONNX 1.23; pytest 9.1.1. Historical training environments are recorded separately.
