# ADR: Authenticated diagnostics and replay

Status: implemented.

I require constant-time bearer-token comparison for prediction, model information, metrics and every forecast path. Replay authentication precedes the replay enablement check. Health probes remain open.

Historical replay stays disabled by default. Registry checks bind the loaded model and metadata bytes; they do not validate caller-supplied features as prospective observations.
