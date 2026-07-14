package llm

import "testing"

func TestTokenUsageAddCost(t *testing.T) {
	positive := 0.01
	second := 0.02
	zero := 0.0

	tests := []struct {
		name      string
		costs     []*float64
		wantNil   bool
		wantValue float64
	}{
		{name: "all absent", costs: []*float64{nil, nil}, wantNil: true},
		{name: "first present", costs: []*float64{nil, &positive}, wantValue: 0.01},
		{name: "later absent", costs: []*float64{&positive, nil}, wantValue: 0.01},
		{name: "sum present", costs: []*float64{&positive, &second}, wantValue: 0.03},
		{name: "explicit zero", costs: []*float64{&zero}, wantValue: 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var usage TokenUsage
			for _, cost := range tt.costs {
				usage.add(TokenUsage{CostUSD: cost})
			}
			if tt.wantNil {
				if usage.CostUSD != nil {
					t.Fatalf("CostUSD = %v, want nil", *usage.CostUSD)
				}
				return
			}
			if usage.CostUSD == nil || *usage.CostUSD != tt.wantValue {
				t.Fatalf("CostUSD = %v, want %v", usage.CostUSD, tt.wantValue)
			}
		})
	}
}

func TestTokenUsageCloneCopiesCost(t *testing.T) {
	cost := 0.01
	original := TokenUsage{CostUSD: &cost}
	cloned := original.clone()

	if cloned.CostUSD == nil || *cloned.CostUSD != cost {
		t.Fatalf("cloned CostUSD = %v, want %v", cloned.CostUSD, cost)
	}
	if cloned.CostUSD == original.CostUSD {
		t.Fatal("cloned CostUSD aliases original")
	}

	*cloned.CostUSD = 0.02
	if *original.CostUSD != 0.01 {
		t.Fatalf("original CostUSD = %v after clone mutation, want 0.01", *original.CostUSD)
	}
}
