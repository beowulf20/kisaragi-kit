// Package tool turns ordinary Go functions into typed LLM tools.
//
// Tool input structs become JSON schemas through reflection, protected toolbox
// calls validate and canonicalize arguments before centralized policy and
// approval hooks, and results are marshaled back to JSON. Schema generation
// covers common Go types and intentionally does not infer richer validation such
// as enums, numeric ranges, regex patterns, string lengths, or recursive schemas.
package tool
