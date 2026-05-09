// Package tool turns ordinary Go functions into typed LLM tools.
//
// Tool input structs become JSON schemas through reflection, calls are decoded
// with unknown-field checks, and results are marshaled back to JSON.
package tool
