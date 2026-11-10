# ADR: Package and asset boundary

Status: current implementation reviewed during portfolio cleanup.

I publish code locally as the chronogrid package and expose chronogrid-ops and chronogrid-registry entry points. Model and dataset assets remain explicit workspace inputs.

An installed code package is not a model-data bundle. I require an explicit valid workspace when an operation needs those assets.
