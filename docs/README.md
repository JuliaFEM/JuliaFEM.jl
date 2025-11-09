---
title: "JuliaFEM Documentation"
description: "Three-tier documentation structure for users, contributors, and researchers"
date: 2025-11-09
author: "Jukka Aho"
categories: ["documentation", "guide"]
keywords: ["juliafem", "finite element", "documentation", "manual"]
type: "index"
---

# JuliaFEM Documentation

Welcome! JuliaFEM documentation is organized into **three manuals** for three different audiences:

---

## 📘 [User Manual](user/) - "Just Get It Done"

**For:** End users, engineers, students who want to run simulations.

**Style:** Simple, practical, step-by-step.

**Contents:**
- Quick start and installation
- Tutorials and examples
- API reference
- Troubleshooting

**Philosophy:** Show me how to solve my problem, skip the lectures.

👉 **[Start Here](user/README.md)** if you want to run simulations.

---

## 🔧 [Contributor Manual](contributor/) - "Show Me the Code"

**For:** Developers, contributors, advanced users who want to extend JuliaFEM.

**Style:** Technical, detailed, design rationale.

**Contents:**
- Testing philosophy
- Code style and architecture
- Performance guidelines
- How to add elements
- CI/CD and git workflow

**Philosophy:** Explain HOW the code works and WHY we made these choices.

👉 **[Start Here](contributor/README.md)** if you want to contribute code.

---

## 📖 [The JuliaFEM Book](book/) - "Let Me Show You How I Think"

**For:** Advanced researchers, theory nerds, those who want to understand deeply. And Jukka.

**Style:** Comprehensive, educational, opinionated, personal.

**Contents:**
- Mathematical foundations (Lagrange basis, contact mechanics, etc.)
- Design philosophy and technical vision
- Strategic mistakes and lessons learned (2015-2019)
- Research directions (nodal assembly, matrix-free, etc.)
- Personal reflections on the journey

**Philosophy:** Mix theory, software design, and personal experience. Teach FEM through implementation.

👉 **[Start Here](book/README.md)** if you love deep dives and want to understand the "why" behind everything.

---

## Quick Navigation

### I want to...

- **Solve a heat transfer problem** → [User Manual](user/)
- **Add a new element type** → [Contributor Manual](contributor/)
- **Understand Lagrange basis functions** → [Book: Lagrange Basis](book/lagrange_basis_functions.md)
- **Learn about testing** → [Contributor: Testing Philosophy](contributor/testing_philosophy.md)
- **See benchmark results** → [Book: Benchmarks](book/benchmarks/)
- **Understand the design philosophy** → [Book: Philosophy](book/)
- **Report a bug** → GitHub Issues
- **Ask a question** → GitHub Discussions

---

## Documentation Philosophy

### Why Three Manuals?

Different readers have different needs:

1. **Users** don't care about implementation details - they just want working code.
2. **Contributors** need technical depth but not necessarily all the theory.
3. **Researchers** (and Jukka) want to understand everything from first principles.

Mixing these audiences in one manual makes it too complex for users and too shallow for researchers.

### Design Principles

- **User Manual:** Optimize for time-to-first-result
- **Contributor Manual:** Optimize for correctness and maintainability
- **Book:** Optimize for understanding and education

### Cross-References

Manuals link to each other when appropriate:
- User manual links to theory when deeper understanding helps
- Contributor manual links to book for design rationale
- Book links to code examples and practical guides

---

## Contributing to Documentation

Documentation improvements are always welcome!

- **User docs:** Fix errors, add examples, improve clarity
- **Contributor docs:** Update for new features, clarify architecture
- **Book:** Add theory, share insights, document research

See [Contributor Manual](contributor/) for guidelines.

---

**License:** MIT (same as code)  
**Questions?** Open an issue or discussion on GitHub
