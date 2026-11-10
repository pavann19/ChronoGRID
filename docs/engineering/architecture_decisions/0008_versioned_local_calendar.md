# ADR: Calendar compatibility

Status: implemented.

I generate nationwide German holidays with the holidays package for the main feature transform. I retain separately versioned compatibility rules and guards for the frozen research adapters.

I do not silently reinterpret a frozen model as using another calendar policy. A wider supported date range does not supply missing training seasons or calibration evidence.
