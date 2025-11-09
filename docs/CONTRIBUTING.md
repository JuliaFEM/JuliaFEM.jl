---
title: "Contributing to JuliaFEM"
description: "Quick start guide for new contributors"
date: 2025-11-09
author: "Jukka Aho"
categories: ["development", "contributing", "getting started"]
keywords: ["juliafem", "contributing", "pull requests", "development"]
audience: "contributors"
level: "beginner"
type: "guide"
---

Thank you for considering contributing to JuliaFEM! 🎉

## Quick Links

- **[Contributor Manual](contributor/README.md)** - Start here for technical details
- **[Coding Standards](contributor/coding_standards.md)** - **REQUIRED** reading for all contributors
- **[Testing Philosophy](contributor/testing_philosophy.md)** - How we test and why

## Quick Start

1. **Fork the repository** on GitHub

2. **Clone your fork:**

   ```bash
   git clone https://github.com/YOUR_USERNAME/JuliaFEM.jl.git
   cd JuliaFEM.jl
   ```

3. **Create a branch:**

   ```bash
   git checkout -b fix-issue-123
   ```

4. **Read the [Coding Standards](contributor/coding_standards.md)** - Critical rules like:
   - ✅ Use `u, v, w` for reference coordinates
   - ❌ Never use Greek letters (ξ, η, ζ) in code
   - ✅ Type-stable code required
   - ✅ Zero allocations in hot paths

5. **Make your changes** following the standards

6. **Run tests:**

   ```bash
   julia --project=. -e 'using Pkg; Pkg.test()'
   ```

7. **Commit with good messages:**

   ```text
   feat(topology): Add Pyr5 pyramid element
   
   - Implement 5-node pyramid reference element
   - Zero-allocation tuple interface
   - Tests for reference coordinates
   
   Closes #123
   ```

8. **Push and create Pull Request**

## Code of Conduct

- Be respectful and constructive
- Focus on the code, not the person
- Welcome newcomers and help them learn
- Ask questions before making assumptions

## What We Look For

✅ **Type-stable code** - Performance depends on it  
✅ **Tests included** - New features need tests  
✅ **Documentation** - Docstrings for exported functions  
✅ **Clean commits** - Logical, well-described changes  
✅ **Follows standards** - Read [coding_standards.md](contributor/coding_standards.md)  

❌ **Type-unstable code** - Will be rejected  
❌ **No tests** - Cannot merge without tests  
❌ **Greek letters in code** - Use u, v, w instead  
❌ **Breaking changes** - Discuss in issue first  

## Getting Help

- **Questions?** Open a GitHub Discussion
- **Bug report?** Open an issue with reproducible example
- **Feature idea?** Open an issue to discuss before implementing
- **Stuck?** Ask in the issue or PR - we're happy to help!

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Ready to contribute?** → Start with [Contributor Manual](contributor/README.md) and [Coding Standards](contributor/coding_standards.md)
