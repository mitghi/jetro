// Cold single-run bench: native Go.
// Each case: parse + execute + serialize. No warmup, no iterations.
// Mirrors jetro-core/examples/bench_cold.rs cases.

package main

import (
	"encoding/json"
	"fmt"
	"sort"
	"time"
)

type Addr struct {
	City string `json:"city"`
	Zip  string `json:"zip"`
}

type User struct {
	Name string `json:"name"`
	Age  int64  `json:"age"`
	Addr Addr   `json:"addr"`
	Tier string `json:"tier"`
}

type Event struct {
	Kind   string `json:"kind"`
	At     string `json:"at,omitempty"`
	Reason string `json:"reason,omitempty"`
}

type Item struct {
	Sku   string  `json:"sku"`
	Qty   int64   `json:"qty"`
	Price float64 `json:"price"`
}

type Record struct {
	ID     int64    `json:"id"`
	User   User     `json:"user"`
	Items  []Item   `json:"items"`
	Tags   []string `json:"tags"`
	Active bool     `json:"active"`
	Score  int64    `json:"score"`
	Status string   `json:"status"`
	Events []Event  `json:"events"`
}

type Wrapper struct {
	Data []Record `json:"data"`
}

func buildDoc(n int) []byte {
	cities := []string{"NYC", "SF", "LA", "Boston", "Seattle", "Austin", "Miami", "Chicago"}
	statuses := []string{"paid", "refunded", "cancelled"}
	tiers := []string{"gold", "silver", "platinum", "bronze"}
	var s []byte
	s = append(s, []byte(`{"data":[`)...)
	for i := 0; i < n; i++ {
		if i > 0 {
			s = append(s, ',')
		}
		city := cities[i%len(cities)]
		status := statuses[i%len(statuses)]
		tier := tiers[(i/3)%len(tiers)]
		itemCount := 3 + (i % 5)
		items := "["
		for k := 0; k < itemCount; k++ {
			if k > 0 {
				items += ","
			}
			items += fmt.Sprintf(`{"sku":"S%d_%d","qty":%d,"price":%g}`,
				i, k, (k+1)*2, float64((i*7+k*13)%200)+9.99)
		}
		items += "]"
		active := "false"
		if i%3 == 0 {
			active = "true"
		}
		var lastKind string
		switch i % 4 {
		case 0:
			lastKind = "delivered"
		case 1:
			lastKind = "shipped"
		case 2:
			lastKind = "refund"
		default:
			lastKind = "placed"
		}
		var events string
		switch lastKind {
		case "refund":
			events = fmt.Sprintf(
				`[{"kind":"placed","at":"2025-04-%02dT10:00:00Z"},{"kind":"refund","reason":"r%d"}]`,
				(i%27)+1, i%5)
		case "placed":
			events = fmt.Sprintf(
				`[{"kind":"placed","at":"2025-04-%02dT10:00:00Z"}]`,
				(i%27)+1)
		default:
			events = fmt.Sprintf(
				`[{"kind":"placed","at":"2025-04-%02dT10:00:00Z"},{"kind":"shipped","at":"2025-04-%02dT08:00:00Z"},{"kind":"%s","at":"2025-04-%02dT17:00:00Z"}]`,
				(i%27)+1, ((i+1)%27)+1, lastKind, ((i+2)%27)+1)
		}
		s = append(s, []byte(fmt.Sprintf(
			`{"id":%d,"user":{"name":"user_%d","age":%d,"addr":{"city":"%s","zip":"%d"},"tier":"%s"},"items":%s,"tags":["t%d","t%d","t%d"],"active":%s,"score":%d,"status":"%s","events":%s}`,
			i, i, 18+(i%60), city, 10000+(i%1000), tier,
			items, i%5, (i+1)%5, (i+2)%5, active, (i*37)%1000, status, events))...)
	}
	s = append(s, []byte("]}")...)
	return s
}

func timeOne(f func()) float64 {
	t0 := time.Now()
	f()
	return float64(time.Since(t0).Microseconds()) / 1000.0
}

func parseFresh(doc []byte) *Wrapper {
	var w Wrapper
	if err := json.Unmarshal(doc, &w); err != nil {
		panic(err)
	}
	return &w
}

func mustJSON(v interface{}) []byte {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return b
}

func printRow(name string, ms float64) {
	fmt.Printf("%-44s go %7.3fms\n", name, ms)
}

func main() {
	n := 8000
	doc := buildDoc(n)
	fmt.Printf("Cold single-run bench (Go) - N=%d  input=%d KB  (no warmup, no iters)\n\n",
		n, len(doc)/1024)

	// 1. filter(active) -> filter(score>200) -> sort(-score) -> take(100)
	//    -> flat_map(items) -> filter(price>50) -> map(qty*price) -> sum
	{
		ms := timeOne(func() {
			w := parseFresh(doc)
			var active []*Record
			for i := range w.Data {
				r := &w.Data[i]
				if r.Active && r.Score > 200 {
					active = append(active, r)
				}
			}
			sort.Slice(active, func(i, j int) bool { return active[i].Score > active[j].Score })
			limit := 100
			if len(active) < limit {
				limit = len(active)
			}
			var total float64
			for _, r := range active[:limit] {
				for _, it := range r.Items {
					if it.Price > 50.0 {
						total += float64(it.Qty) * it.Price
					}
				}
			}
			_ = mustJSON(total)
		})
		printRow("active top-100 expensive-item revenue", ms)
	}

	// 2. flat_map(items) -> sort(-price) -> take(30) -> map({sku,price})
	{
		ms := timeOne(func() {
			w := parseFresh(doc)
			type pair struct {
				sku   string
				price float64
			}
			var all []pair
			for i := range w.Data {
				for _, it := range w.Data[i].Items {
					all = append(all, pair{it.Sku, it.Price})
				}
			}
			sort.Slice(all, func(i, j int) bool { return all[i].price > all[j].price })
			limit := 30
			if len(all) < limit {
				limit = len(all)
			}
			type out struct {
				Sku   string  `json:"sku"`
				Price float64 `json:"price"`
			}
			outs := make([]out, 0, limit)
			for _, p := range all[:limit] {
				outs = append(outs, out{p.sku, p.price})
			}
			_ = mustJSON(outs)
		})
		printRow("flatmap+sort all-items+take+project", ms)
	}

	// 3. sort(-score) -> skip(200) -> take(50) -> map({id, city, score})
	{
		ms := timeOne(func() {
			w := parseFresh(doc)
			sorted := make([]*Record, len(w.Data))
			for i := range w.Data {
				sorted[i] = &w.Data[i]
			}
			sort.Slice(sorted, func(i, j int) bool { return sorted[i].Score > sorted[j].Score })
			type out struct {
				ID    int64  `json:"id"`
				City  string `json:"city"`
				Score int64  `json:"score"`
			}
			start, end := 200, 250
			if end > len(sorted) {
				end = len(sorted)
			}
			if start > len(sorted) {
				start = len(sorted)
			}
			page := sorted[start:end]
			outs := make([]out, 0, len(page))
			for _, r := range page {
				outs = append(outs, out{r.ID, r.User.Addr.City, r.Score})
			}
			_ = mustJSON(outs)
		})
		printRow("sort+skip+take+project", ms)
	}

	// 4. filter(active) -> flat_map(tags) -> unique
	{
		ms := timeOne(func() {
			w := parseFresh(doc)
			var tags []string
			for i := range w.Data {
				r := &w.Data[i]
				if r.Active {
					tags = append(tags, r.Tags...)
				}
			}
			sort.Strings(tags)
			tags = dedupStrings(tags)
			_ = mustJSON(tags)
		})
		printRow("filter+flatmap-tags+unique", ms)
	}

	// 5. flat_map(items) -> filter(price>100) -> map(qty*price) -> sum
	{
		ms := timeOne(func() {
			w := parseFresh(doc)
			var total float64
			for i := range w.Data {
				for _, it := range w.Data[i].Items {
					if it.Price > 100.0 {
						total += float64(it.Qty) * it.Price
					}
				}
			}
			_ = mustJSON(total)
		})
		printRow("flatmap+filter+map-arith+sum", ms)
	}

	// 6. filter(active) -> sort(-score) -> take(50) -> map(fstring)
	{
		ms := timeOne(func() {
			w := parseFresh(doc)
			var active []*Record
			for i := range w.Data {
				r := &w.Data[i]
				if r.Active {
					active = append(active, r)
				}
			}
			sort.Slice(active, func(i, j int) bool { return active[i].Score > active[j].Score })
			limit := 50
			if len(active) < limit {
				limit = len(active)
			}
			outs := make([]string, 0, limit)
			for _, r := range active[:limit] {
				outs = append(outs, fmt.Sprintf("#%d %s (%s) score=%d",
					r.ID, r.User.Name, r.User.Addr.City, r.Score))
			}
			_ = mustJSON(outs)
		})
		printRow("filter+sort+take+fstring", ms)
	}

	// 7. filter(score>700) -> flat_map(items) -> map(price) -> avg
	{
		ms := timeOne(func() {
			w := parseFresh(doc)
			var prices []float64
			for i := range w.Data {
				if w.Data[i].Score > 700 {
					for _, it := range w.Data[i].Items {
						prices = append(prices, it.Price)
					}
				}
			}
			var avg float64
			if len(prices) > 0 {
				var sum float64
				for _, p := range prices {
					sum += p
				}
				avg = sum / float64(len(prices))
			}
			_ = mustJSON(avg)
		})
		printRow("filter+flatmap+avg", ms)
	}

	// 8. sort(-score) -> take(20) -> map({id, city, total})
	{
		ms := timeOne(func() {
			w := parseFresh(doc)
			sorted := make([]*Record, len(w.Data))
			for i := range w.Data {
				sorted[i] = &w.Data[i]
			}
			sort.Slice(sorted, func(i, j int) bool { return sorted[i].Score > sorted[j].Score })
			type out struct {
				ID    int64   `json:"id"`
				City  string  `json:"city"`
				Total float64 `json:"total"`
			}
			limit := 20
			if len(sorted) < limit {
				limit = len(sorted)
			}
			outs := make([]out, 0, limit)
			for _, r := range sorted[:limit] {
				var total float64
				for _, it := range r.Items {
					total += float64(it.Qty) * it.Price
				}
				outs = append(outs, out{r.ID, r.User.Addr.City, total})
			}
			_ = mustJSON(outs)
		})
		printRow("sort+take+nested-computed-projection", ms)
	}

	// 9. filter(active) -> filter(score>500) -> flat_map(items)
	//    -> filter(price>75) -> filter(qty>2) -> len
	{
		ms := timeOne(func() {
			w := parseFresh(doc)
			count := 0
			for i := range w.Data {
				r := &w.Data[i]
				if r.Active && r.Score > 500 {
					for _, it := range r.Items {
						if it.Price > 75.0 && it.Qty > 2 {
							count++
						}
					}
				}
			}
			_ = mustJSON(count)
		})
		printRow("5-stage filter chain + count", ms)
	}

	// 10. count_by(active)
	{
		ms := timeOne(func() {
			w := parseFresh(doc)
			var t, f uint64
			for i := range w.Data {
				if w.Data[i].Active {
					t++
				} else {
					f++
				}
			}
			_ = mustJSON(map[string]uint64{"true": t, "false": f})
		})
		printRow("count_by(active) / group_by+map", ms)
	}

	// 11. sort(-score) -> take(300) -> map(zip) -> unique
	{
		ms := timeOne(func() {
			w := parseFresh(doc)
			sorted := make([]*Record, len(w.Data))
			for i := range w.Data {
				sorted[i] = &w.Data[i]
			}
			sort.Slice(sorted, func(i, j int) bool { return sorted[i].Score > sorted[j].Score })
			limit := 300
			if len(sorted) < limit {
				limit = len(sorted)
			}
			zips := make([]string, 0, limit)
			for _, r := range sorted[:limit] {
				zips = append(zips, r.User.Addr.Zip)
			}
			sort.Strings(zips)
			zips = dedupStrings(zips)
			_ = mustJSON(zips)
		})
		printRow("sort+take+map+unique (top-300 zips)", ms)
	}

	// 12. flat_map(items) -> map(price) -> unique -> len
	{
		ms := timeOne(func() {
			w := parseFresh(doc)
			var prices []int64
			for i := range w.Data {
				for _, it := range w.Data[i].Items {
					prices = append(prices, int64(it.Price*100.0))
				}
			}
			sort.Slice(prices, func(i, j int) bool { return prices[i] < prices[j] })
			n := 0
			for i, p := range prices {
				if i == 0 || p != prices[i-1] {
					n++
				}
			}
			_ = mustJSON(n)
		})
		printRow("flatmap+map+unique+len (all prices)", ms)
	}

	fmt.Println("\n--- v2 cases ---\n")

	// 13. filter+map+sum
	{
		ms := timeOne(func() {
			w := parseFresh(doc)
			var s int64
			for i := range w.Data {
				if w.Data[i].Active {
					s += w.Data[i].Score
				}
			}
			_ = mustJSON(s)
		})
		printRow("filter+map+sum", ms)
	}

	// 14. flat_map+filter+count
	{
		ms := timeOne(func() {
			w := parseFresh(doc)
			n := 0
			for i := range w.Data {
				for _, it := range w.Data[i].Items {
					if it.Price > 50.0 {
						n++
					}
				}
			}
			_ = mustJSON(n)
		})
		printRow("flat_map+filter+count", ms)
	}

	// 15. filter+flat_map+map+sum
	{
		ms := timeOne(func() {
			w := parseFresh(doc)
			var s float64
			for i := range w.Data {
				r := &w.Data[i]
				if r.Active {
					for _, it := range r.Items {
						s += float64(it.Qty) * it.Price
					}
				}
			}
			_ = mustJSON(s)
		})
		printRow("filter+flat_map+map+sum", ms)
	}

	// 16. sort_by+take+map (top10)
	{
		ms := timeOne(func() {
			w := parseFresh(doc)
			idx := make([]*Record, len(w.Data))
			for i := range w.Data {
				idx[i] = &w.Data[i]
			}
			sort.Slice(idx, func(i, j int) bool { return idx[i].Score > idx[j].Score })
			type top struct {
				ID    int64  `json:"id"`
				Name  string `json:"name"`
				Score int64  `json:"score"`
			}
			limit := 10
			if len(idx) < limit {
				limit = len(idx)
			}
			outs := make([]top, 0, limit)
			for _, r := range idx[:limit] {
				outs = append(outs, top{r.ID, r.User.Name, r.Score})
			}
			_ = mustJSON(outs)
		})
		printRow("sort_by+take+map (top10)", ms)
	}

	// 17. map+unique cities
	{
		ms := timeOne(func() {
			w := parseFresh(doc)
			seen := map[string]struct{}{}
			for i := range w.Data {
				seen[w.Data[i].User.Addr.City] = struct{}{}
			}
			cities := make([]string, 0, len(seen))
			for c := range seen {
				cities = append(cities, c)
			}
			sort.Strings(cities)
			_ = mustJSON(cities)
		})
		printRow("map+unique (cities)", ms)
	}

	// 18. deep projection
	{
		ms := timeOne(func() {
			w := parseFresh(doc)
			type out struct {
				ID        int64   `json:"id"`
				City      string  `json:"city"`
				ItemCount int     `json:"item_count"`
				Total     float64 `json:"total"`
			}
			outs := make([]out, 0, len(w.Data))
			for i := range w.Data {
				r := &w.Data[i]
				var total float64
				for _, it := range r.Items {
					total += float64(it.Qty) * it.Price
				}
				outs = append(outs, out{r.ID, r.User.Addr.City, len(r.Items), total})
			}
			_ = mustJSON(outs)
		})
		printRow("map (deep projection)", ms)
	}

	// 19. f-string per row
	{
		ms := timeOne(func() {
			w := parseFresh(doc)
			outs := make([]string, 0, len(w.Data))
			for i := range w.Data {
				r := &w.Data[i]
				outs = append(outs, fmt.Sprintf("#%d %s (%s) $%d",
					r.ID, r.User.Name, r.User.Addr.City, r.Score))
			}
			_ = mustJSON(outs)
		})
		printRow("map f-string", ms)
	}

	// 20. flat_map+map all prices
	{
		ms := timeOne(func() {
			w := parseFresh(doc)
			var prices []float64
			for i := range w.Data {
				for _, it := range w.Data[i].Items {
					prices = append(prices, it.Price)
				}
			}
			_ = mustJSON(prices)
		})
		printRow("flat_map+map (all prices)", ms)
	}

	// 21. filter+first
	{
		ms := timeOne(func() {
			w := parseFresh(doc)
			var firstID *int64
			for i := range w.Data {
				if w.Data[i].Score > 900 {
					id := w.Data[i].ID
					firstID = &id
					break
				}
			}
			_ = mustJSON(firstID)
		})
		printRow("filter+first", ms)
	}

	// 22. skip+take+map pagination
	{
		ms := timeOne(func() {
			w := parseFresh(doc)
			start, end := 100, 120
			if end > len(w.Data) {
				end = len(w.Data)
			}
			if start > len(w.Data) {
				start = len(w.Data)
			}
			type page struct {
				ID int64 `json:"id"`
			}
			outs := make([]page, 0, end-start)
			for i := start; i < end; i++ {
				outs = append(outs, page{w.Data[i].ID})
			}
			_ = mustJSON(outs)
		})
		printRow("skip+take+map (pagination)", ms)
	}

	// 23. filter+map+avg
	{
		ms := timeOne(func() {
			w := parseFresh(doc)
			var sum int64
			n := 0
			for i := range w.Data {
				if w.Data[i].Active {
					sum += w.Data[i].Score
					n++
				}
			}
			var avg float64
			if n > 0 {
				avg = float64(sum) / float64(n)
			}
			_ = mustJSON(avg)
		})
		printRow("filter+map+avg", ms)
	}

	fmt.Println("\n--- README showcase ---\n")

	// 24. README showcase: 3-stage filter -> sort -> take -> projection
	//     including nested per-row sum and a switch over events[-1].kind.
	{
		ms := timeOne(func() {
			w := parseFresh(doc)
			type lastEvent struct {
				State  string `json:"state"`
				At     string `json:"at,omitempty"`
				Reason string `json:"reason,omitempty"`
			}
			type out struct {
				ID        int64     `json:"id"`
				Who       string    `json:"who"`
				Tier      string    `json:"tier"`
				ScoreVal  int64     `json:"score_val"`
				Label     string    `json:"label"`
				LineTotal float64   `json:"line_total"`
				LastEvent lastEvent `json:"last_event"`
			}
			var hits []*Record
			for i := range w.Data {
				r := &w.Data[i]
				if r.Status != "paid" {
					continue
				}
				if r.Score < 500 {
					continue
				}
				if r.User.Tier != "gold" && r.User.Tier != "platinum" {
					continue
				}
				hits = append(hits, r)
			}
			sort.Slice(hits, func(i, j int) bool { return hits[i].Score > hits[j].Score })
			limit := 50
			if len(hits) < limit {
				limit = len(hits)
			}
			outs := make([]out, 0, limit)
			for _, r := range hits[:limit] {
				var lt float64
				for _, it := range r.Items {
					lt += float64(it.Qty) * it.Price
				}
				var le lastEvent
				if len(r.Events) > 0 {
					e := r.Events[len(r.Events)-1]
					switch e.Kind {
					case "delivered":
						le = lastEvent{State: "ok", At: e.At}
					case "shipped":
						le = lastEvent{State: "moving", At: e.At}
					case "refund":
						le = lastEvent{State: "refund", Reason: e.Reason}
					default:
						le = lastEvent{State: "unknown"}
					}
				} else {
					le = lastEvent{State: "unknown"}
				}
				outs = append(outs, out{
					ID:        r.ID,
					Who:       r.User.Name,
					Tier:      r.User.Tier,
					ScoreVal:  r.Score,
					Label:     fmt.Sprintf("order %d: %s (%s) score %d", r.ID, r.User.Name, r.User.Tier, r.Score),
					LineTotal: lt,
					LastEvent: le,
				})
			}
			_ = mustJSON(outs)
		})
		printRow("README showcase (3-filter+sort+take+match)", ms)
	}
}

func dedupStrings(s []string) []string {
	if len(s) == 0 {
		return s
	}
	out := s[:1]
	for _, v := range s[1:] {
		if v != out[len(out)-1] {
			out = append(out, v)
		}
	}
	return out
}
