# ADR: SQLite recovery

Status: current implementation reviewed during portfolio cleanup.

I use SQLite transactions and explicit journal verification. I preserve a failed journal for investigation and work on a backup when testing recovery.

I do not repair a journal by deleting identity or feedback records. A successful recovery test applies to its isolated fixture, not to every filesystem or deployment.
