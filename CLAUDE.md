# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

## Commands

## Tools & Setup
- Line length: 88 chars
- always use serena for code searching when possible (its an MCP)
- always use context7 mcp or the skill find-docs when looking up documentation for libraries

## Code Conventions

### Hard Rules
- **MUST** include docstrings (with args, returns, raises) for all public functions, classes, and methods
- **MUST** utilize SOLID coding principles
- **MUST** use design patterns (https://refactoring.guru/design-patterns/python) when appropriate.
- **MUST** Not implement leaky abstractions
- **ALWAYS** prefer quick returns or continues over deep nesting of the happy path 
    - i.e. 
        if something is not True:                <--- prefer this always
            return
      vs
        if something:
            // do stuff
- **NO magic numbers** — use named constants only
- **NO emoji or unicode emoji substitutes** (e.g. checkmarks, crosses) — in code AND in Gainsley's output
- **NO mutable default arguments**
- **NO bare `except:`** — catch specific exceptions
- **NO `print` for errors** — use `logger.error`
- **NO secrets in code** — `.env` only; ensure `.env` and test output dirs are in `.gitignore`
- **NO logging of sensitive data** (passwords, tokens, PII)

### Design
- Max 5 parameters per function (`__init__` excluded)
- Dependency injection for complex dependencies; classes must be mockable
- Keep `__init__` simple; no complex logic

### Code Review
- **MUST** Check for leaky abstractions and poor code design
- **MUST** Pay special attention to failure modes, data races
- **ALWAYS** Check for failure modes in distributed systems (i.e. races, lost writes, partial writes, etc)

### Testing
#### Unit Testing
- Mock all external dependencies (APIs, DBs, filesystem)
- Save test files before running; never delete them
#### Functional Testing
- Always generate a suite of functional tests that test the expected behavior of the application as if it
  is a black box. These tests should focus on testing behavior end to end to validate the requirements
  of the system

### Commits
- No commented-out code, debug prints, or credentials
