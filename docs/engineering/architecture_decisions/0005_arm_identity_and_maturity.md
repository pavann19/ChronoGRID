# ADR: Model identity and label maturity

Status: current implementation reviewed during portfolio cleanup.

I bind an evaluation arm to its registered model hash and settle actual observations only after the configured maturity boundary. Missing and not-yet-mature labels remain distinct.

I do not treat missing outcomes as successful coverage or permit a later model identity to replace an already registered arm.
