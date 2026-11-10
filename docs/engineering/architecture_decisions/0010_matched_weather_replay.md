# ADR: Preserved weather replay

Status: current implementation reviewed during portfolio cleanup.

I retain the historical weather replay as a checksum-verified compressed fixture. Replay tests unpack it only after verifying the archive and compare reconstructed features and predictions against its saved evidence.

The old replay remains a separate historical experiment. It does not replace the corrected benchmark, provide a fresh holdout or authorize model promotion.
