package main

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/beowulf20/kisaragi-kit/pkg/llm"
)

type supplier struct {
	ID             string `json:"id"`
	UnitPriceCents int    `json:"unit_price_cents"`
	ShippingCents  int    `json:"shipping_cents"`
	Stock          int    `json:"stock"`
	DeliveryDays   int    `json:"delivery_days"`
}

var catalog = []supplier{
	{ID: "north", UnitPriceCents: 2400, ShippingCents: 1200, Stock: 40, DeliveryDays: 5},
	{ID: "south", UnitPriceCents: 2300, ShippingCents: 2400, Stock: 30, DeliveryDays: 9},
	{ID: "west", UnitPriceCents: 2500, ShippingCents: 0, Stock: 20, DeliveryDays: 3},
}

// Pacing simulates local demo backend latency, independently of model latency.
type demoTools struct{ delay time.Duration }

func (d demoTools) step(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if d.delay == 0 {
		return nil
	}
	timer := time.NewTimer(d.delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return ctx.Err()
	}
}

type catalogInput struct {
	Category string `json:"category" description:"Demo category: sensor"`
}
type quoteInput struct {
	Quantity int `json:"quantity" description:"Sensor quantity, from 1 to 100"`
}
type deliveryInput struct {
	MaxDays int `json:"max_days" description:"Delivery deadline in days, from 1 to 60"`
}

func validateQuantity(quantity int) error {
	if quantity < 1 || quantity > 100 {
		return errors.New("quantity must be between 1 and 100")
	}
	return nil
}
func validateDeadline(days int) error {
	if days < 1 || days > 60 {
		return errors.New("delivery deadline must be between 1 and 60 days")
	}
	return nil
}

func (d demoTools) lookupCatalog(ctx context.Context, input catalogInput) ([]supplier, error) {
	if strings.ToLower(strings.TrimSpace(input.Category)) != "sensor" {
		return nil, errors.New("the demo catalog contains only the sensor category")
	}
	if err := d.step(ctx); err != nil {
		return nil, err
	}
	llm.ReportMilestone(ctx, "Verified 3 suppliers in the local demo sensor catalog")
	return append([]supplier(nil), catalog...), nil
}

type inventoryCheck struct {
	SupplierID string `json:"supplier_id"`
	Stock      int    `json:"stock"`
	Sufficient bool   `json:"sufficient"`
}

func (d demoTools) checkInventory(ctx context.Context, input quoteInput) ([]inventoryCheck, error) {
	if err := validateQuantity(input.Quantity); err != nil {
		return nil, err
	}
	var checks []inventoryCheck
	for _, offer := range catalog {
		if err := d.step(ctx); err != nil {
			return nil, err
		}
		check := inventoryCheck{SupplierID: offer.ID, Stock: offer.Stock, Sufficient: offer.Stock >= input.Quantity}
		checks = append(checks, check)
		llm.ReportMilestone(ctx, fmt.Sprintf("Checked %s demo stock: %d available for %d requested; sufficient=%t", offer.ID, offer.Stock, input.Quantity, check.Sufficient))
	}
	return checks, nil
}

type deliveryCheck struct {
	SupplierID     string `json:"supplier_id"`
	Days           int    `json:"days"`
	WithinDeadline bool   `json:"within_deadline"`
}

func (d demoTools) checkDelivery(ctx context.Context, input deliveryInput) ([]deliveryCheck, error) {
	if err := validateDeadline(input.MaxDays); err != nil {
		return nil, err
	}
	var checks []deliveryCheck
	for _, offer := range catalog {
		if err := d.step(ctx); err != nil {
			return nil, err
		}
		check := deliveryCheck{SupplierID: offer.ID, Days: offer.DeliveryDays, WithinDeadline: offer.DeliveryDays <= input.MaxDays}
		checks = append(checks, check)
		llm.ReportMilestone(ctx, fmt.Sprintf("Checked %s demo delivery: %d days against %d-day deadline; eligible=%t", offer.ID, offer.DeliveryDays, input.MaxDays, check.WithinDeadline))
	}
	return checks, nil
}

type quote struct {
	SupplierID   string `json:"supplier_id"`
	TotalCents   int    `json:"total_cents"`
	DeliveryDays int    `json:"delivery_days"`
}

func quoteFor(offer supplier, quantity int) quote {
	return quote{SupplierID: offer.ID, TotalCents: quantity*offer.UnitPriceCents + offer.ShippingCents, DeliveryDays: offer.DeliveryDays}
}

func (d demoTools) calculateQuotes(ctx context.Context, input quoteInput) ([]quote, error) {
	if err := validateQuantity(input.Quantity); err != nil {
		return nil, err
	}
	var quotes []quote
	for _, offer := range catalog {
		if err := d.step(ctx); err != nil {
			return nil, err
		}
		if offer.Stock < input.Quantity {
			llm.ReportMilestone(ctx, fmt.Sprintf("Excluded %s demo quote: stock %d is below quantity %d", offer.ID, offer.Stock, input.Quantity))
			continue
		}
		q := quoteFor(offer, input.Quantity)
		quotes = append(quotes, q)
		llm.ReportMilestone(ctx, fmt.Sprintf("Calculated %s demo delivered quote: %d cents for %d sensors", offer.ID, q.TotalCents, input.Quantity))
	}
	return quotes, nil
}

func cheapestEligible(quantity, maxDays int) *quote {
	var best *quote
	for _, offer := range catalog {
		if offer.Stock < quantity || offer.DeliveryDays > maxDays {
			continue
		}
		candidate := quoteFor(offer, quantity)
		if best == nil || candidate.TotalCents < best.TotalCents {
			best = &candidate
		}
	}
	return best
}

type scenarioInput struct {
	Quantities []int `json:"quantities" description:"One to five distinct quantities, each from 1 to 100"`
	MaxDays    int   `json:"max_days" description:"Delivery deadline, from 1 to 60 days"`
}
type scenarioResult struct {
	Quantity int    `json:"quantity"`
	Cheapest *quote `json:"cheapest_eligible"`
}

func (d demoTools) evaluateScenarios(ctx context.Context, input scenarioInput) ([]scenarioResult, error) {
	if err := validateDeadline(input.MaxDays); err != nil {
		return nil, err
	}
	if len(input.Quantities) < 1 || len(input.Quantities) > 5 {
		return nil, errors.New("provide one to five distinct quantities")
	}
	seen := make(map[int]bool)
	for _, quantity := range input.Quantities {
		if err := validateQuantity(quantity); err != nil {
			return nil, err
		}
		if seen[quantity] {
			return nil, errors.New("scenario quantities must be distinct")
		}
		seen[quantity] = true
	}
	var results []scenarioResult
	for _, quantity := range input.Quantities {
		if err := d.step(ctx); err != nil {
			return nil, err
		}
		best := cheapestEligible(quantity, input.MaxDays)
		results = append(results, scenarioResult{Quantity: quantity, Cheapest: best})
		message := fmt.Sprintf("Evaluated demo quantity %d under %d-day deadline: no eligible supplier", quantity, input.MaxDays)
		if best != nil {
			message = fmt.Sprintf("Evaluated demo quantity %d under %d-day deadline: %s at %d cents", quantity, input.MaxDays, best.SupplierID, best.TotalCents)
		}
		llm.ReportMilestone(ctx, message)
	}
	return results, nil
}

type auditInput struct {
	SupplierID string `json:"supplier_id" description:"Chosen demo supplier ID"`
	Quantity   int    `json:"quantity" description:"Primary order quantity, from 1 to 100"`
	MaxDays    int    `json:"max_days" description:"Delivery deadline, from 1 to 60 days"`
	TotalCents int    `json:"total_cents" description:"Exact proposed delivered total from calculate_quotes"`
}
type auditResult struct {
	StockOK            bool   `json:"stock_ok"`
	DeliveryOK         bool   `json:"delivery_ok"`
	TotalOK            bool   `json:"total_ok"`
	CheapestOK         bool   `json:"cheapest_ok"`
	ExpectedTotalCents int    `json:"expected_total_cents"`
	CheapestSupplierID string `json:"cheapest_supplier_id"`
	Approved           bool   `json:"approved"`
}

func (d demoTools) auditRecommendation(ctx context.Context, input auditInput) (auditResult, error) {
	if err := validateQuantity(input.Quantity); err != nil {
		return auditResult{}, err
	}
	if err := validateDeadline(input.MaxDays); err != nil {
		return auditResult{}, err
	}
	if input.TotalCents < 0 {
		return auditResult{}, errors.New("proposed total cannot be negative")
	}
	var chosen *supplier
	for i := range catalog {
		if catalog[i].ID == input.SupplierID {
			chosen = &catalog[i]
			break
		}
	}
	if chosen == nil {
		return auditResult{}, errors.New("unknown demo supplier")
	}
	var result auditResult
	if err := d.step(ctx); err != nil {
		return result, err
	}
	result.StockOK = chosen.Stock >= input.Quantity
	llm.ReportMilestone(ctx, fmt.Sprintf("Audited %s demo stock for %d sensors; passed=%t", chosen.ID, input.Quantity, result.StockOK))
	if err := d.step(ctx); err != nil {
		return result, err
	}
	result.DeliveryOK = chosen.DeliveryDays <= input.MaxDays
	llm.ReportMilestone(ctx, fmt.Sprintf("Audited %s demo delivery against %d-day deadline; passed=%t", chosen.ID, input.MaxDays, result.DeliveryOK))
	if err := d.step(ctx); err != nil {
		return result, err
	}
	result.ExpectedTotalCents = quoteFor(*chosen, input.Quantity).TotalCents
	result.TotalOK = input.TotalCents == result.ExpectedTotalCents
	if best := cheapestEligible(input.Quantity, input.MaxDays); best != nil {
		result.CheapestSupplierID = best.SupplierID
		result.CheapestOK = result.StockOK && result.DeliveryOK && result.ExpectedTotalCents == best.TotalCents
		if result.CheapestOK {
			result.CheapestSupplierID = chosen.ID // Either eligible supplier is valid in a price tie.
		}
	}
	result.Approved = result.StockOK && result.DeliveryOK && result.TotalOK && result.CheapestOK
	llm.ReportMilestone(ctx, fmt.Sprintf("Audited %s demo total and selection: expected=%d cents; total_ok=%t cheapest_ok=%t approved=%t", chosen.ID, result.ExpectedTotalCents, result.TotalOK, result.CheapestOK, result.Approved))
	return result, nil
}
