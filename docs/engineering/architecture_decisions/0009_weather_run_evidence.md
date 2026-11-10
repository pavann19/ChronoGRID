# ADR: Weather run identity

Status: current implementation reviewed during portfolio cleanup.

I keep the requested model run, source response and issue-time fields identifiable in the separate weather adapter. The corrected benchmark itself uses lagged persistence weather.

A retrieved historical response is not proof that the same revision was available at the original gate. Archived NWP vintages and independently timestamped availability remain future work.
