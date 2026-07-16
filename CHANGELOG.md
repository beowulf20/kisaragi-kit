# Changelog

## Unreleased

### Features

* **llm:** add provider-neutral message guardrails and bounded-run controls
* **guardrail:** add attachable system-prompt leakage matcher
* **guardrail:** add attachable tool-metadata leakage matcher
* **tool:** add central policy hooks and validated approval flow

## [1.9.1](https://github.com/beowulf20/kisaragi-kit/compare/v1.9.0...v1.9.1) (2026-07-16)


### Bug Fixes

* **llm:** preserve output on runtime failures ([58b413d](https://github.com/beowulf20/kisaragi-kit/commit/58b413d9cb02d346b168ed46c64947bcb1f39ad9))
* **llm:** preserve output on runtime failures ([9754f6b](https://github.com/beowulf20/kisaragi-kit/commit/9754f6b9b6a52ca5e99f4ac76bae755ece94d3d0))

## [1.9.0](https://github.com/beowulf20/kisaragi-kit/compare/v1.8.1...v1.9.0) (2026-07-14)


### Features

* **example:** show usage cost breakdown ([3984c16](https://github.com/beowulf20/kisaragi-kit/commit/3984c1642839ea9d282a0dd568a399d38f5b1f10))
* **llm:** add configurable safety guardrails ([04a3945](https://github.com/beowulf20/kisaragi-kit/commit/04a3945644fa5b623d413fd1b1665493727cd931))
* **llm:** preserve provider-reported usage cost ([bda197c](https://github.com/beowulf20/kisaragi-kit/commit/bda197c24bc751e026e546c5f886df289683fdf8))
* **openai:** capture provider-reported usage cost ([902ee88](https://github.com/beowulf20/kisaragi-kit/commit/902ee881558b0359dc3156b107cd94265272bcd4))
* **openrouter:** add typed provider routing controls ([a6526dd](https://github.com/beowulf20/kisaragi-kit/commit/a6526dd52a668df2f8fb00355afc7c37485e3d98))

## [1.8.1](https://github.com/beowulf20/kisaragi-kit/compare/v1.8.0...v1.8.1) (2026-07-01)


### Bug Fixes

* **llm:** preserve abort transcripts ([f02724b](https://github.com/beowulf20/kisaragi-kit/commit/f02724b00e6789ab1600773054444009bd5590a7))

## [1.8.0](https://github.com/beowulf20/kisaragi-kit/compare/v1.7.0...v1.8.0) (2026-06-30)


### Features

* **agent:** forward lifecycle hooks ([cf63592](https://github.com/beowulf20/kisaragi-kit/commit/cf635921d0aeedc001750c6560efcf5a3eb2f34c))
* **llm:** add lifecycle event hooks ([168f0f2](https://github.com/beowulf20/kisaragi-kit/commit/168f0f25f0fcc363e3fb06ec19cdc85284345706))
* **openai:** stream reasoning deltas ([a4c7be7](https://github.com/beowulf20/kisaragi-kit/commit/a4c7be7ed7d50ed3b22f077cefba796a430a4a11))

## [1.7.0](https://github.com/beowulf20/kisaragi-kit/compare/v1.6.0...v1.7.0) (2026-06-14)


### Features

* **openai:** allow chat completion extra fields ([b34dea0](https://github.com/beowulf20/kisaragi-kit/commit/b34dea0978a6937fc28fb28f07c32aad70c1d731))

## [1.6.0](https://github.com/beowulf20/kisaragi-kit/compare/v1.5.0...v1.6.0) (2026-06-14)


### Features

* **llm:** add typed reasoning effort setting ([463d901](https://github.com/beowulf20/kisaragi-kit/commit/463d9014bc4634679d8c42f95073f22b0db2b8c6))

## [1.5.0](https://github.com/beowulf20/kisaragi-kit/compare/v1.4.0...v1.5.0) (2026-06-14)


### Features

* **llm:** add caller context support ([fa16c9d](https://github.com/beowulf20/kisaragi-kit/commit/fa16c9ddda2add5c1750a79ed2c0e8dfc6cb816b))


### Bug Fixes

* **llm:** stabilize tools and usage reporting ([61e6900](https://github.com/beowulf20/kisaragi-kit/commit/61e69002c0dc3ffb6b946dc6f4daa6a2993d117e))

## [1.4.0](https://github.com/beowulf20/kisaragi-kit/compare/v1.3.0...v1.4.0) (2026-05-29)


### Features

* add tool approval hooks ([80eb02d](https://github.com/beowulf20/kisaragi-kit/commit/80eb02dd7370da9e8faeca177685570bfff004f1))

## [1.3.0](https://github.com/beowulf20/kisaragi-kit/compare/v1.2.0...v1.3.0) (2026-05-16)


### Features

* add core llm usage hooks ([1f58185](https://github.com/beowulf20/kisaragi-kit/commit/1f581850f7e48799b5d6e780aec8178c18be3029))

## [1.2.0](https://github.com/beowulf20/kisaragi-kit/compare/v1.1.0...v1.2.0) (2026-05-12)


### Features

* **llm:** add error hooks ([2e128a0](https://github.com/beowulf20/kisaragi-kit/commit/2e128a00c7345a0e0480f05ef30723dd625822d2))

## [1.1.0](https://github.com/beowulf20/kisaragi-kit/compare/v1.0.0...v1.1.0) (2026-05-12)


### Features

* **llm:** add configurable completion controls ([b8110c6](https://github.com/beowulf20/kisaragi-kit/commit/b8110c6a6f2118997a375933bb01cf56fe954eee))

## 1.0.0 (2026-05-09)


### Features

* initial release ([8bb6751](https://github.com/beowulf20/kisaragi-kit/commit/8bb67510a9daae4be9b54e853f76fc72081aa6e1))
