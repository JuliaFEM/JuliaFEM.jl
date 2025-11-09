---
title: "JuliaFEM Contributor Manual"
description: "Technical guide for developers and contributors"
date: 2025-11-09
author: "Jukka Aho"
categories: ["development", "contributor guide"]
keywords: ["juliafem", "development", "architecture", "testing", "performance"]
audience: "developers"
level: "advanced"
type: "manual"
---

**Audience:** Developers, contributors, advanced users who want to extend or modify JuliaFEM.

This manual is **technical and detailed** - it explains HOW the code works and WHY we made certain design choices.

## What's Here

- **Testing Philosophy:** How and why we test
- **Coding Standards:** Required conventions for all contributions (variable names, types, performance)
- **Architecture:** Module structure, data flow, key abstractions
- **Performance:** Zero-allocation design, profiling, benchmarking
- **Adding Elements:** How to implement new element types
- **CI/CD:** Continuous integration, releases, versioning
- **Git Workflow:** Branching, commits, pull requests

## What's NOT Here

- User tutorials (see `docs/user/` for that)
- Deep mathematical theory (see `docs/book/` for that)
- "How do I solve problem X?" (that's user docs)

## Philosophy

**"Show me the code AND tell me why."**

We assume you:

- Know Julia reasonably well
- Understand FEM basics
- Want to add features or fix bugs
- Care about performance and correctness
- Need to understand design rationale

## Before Contributing

1. Read [Testing Philosophy](testing_philosophy.md)
2. Follow [Coding Standards](coding_standards.md) - **REQUIRED** for all contributions
3. Understand [Architecture](architecture.md)
4. Check [Performance Guidelines](performance.md)
5. Review [Git Workflow](git_workflow.md)

## Key Principles

- **Type stability:** No `Any`, no `Dict` without types
- **Zero allocations:** Hot paths should allocate nothing
- **Immutability:** Prefer `struct` over `mutable struct`
- **Composition:** Use tuples and free functions, not OOP hierarchies
- **Explicit:** No magic, user knows what happens
- **Test first:** Write tests before fixing bugs

---

**Start here:** [Testing Philosophy](testing_philosophy.md) | [Architecture Overview](architecture.md)
