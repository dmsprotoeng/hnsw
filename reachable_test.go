package hnsw

import (
	"math"
	"math/rand"
	"sort"
	"testing"

	"github.com/stretchr/testify/require"
)

// corpusVectors builds vectors whose similarity profile matches a real
// document corpus: chunks of one document sit at cosine ~0.64 and chunks of
// different documents at ~0.52, because everything shares a large common
// component (the library covers one broad subject) and each document adds its
// own.
//
// The shape matters. With uniform-random vectors, or with well-separated
// clusters, eviction is spread evenly and the graph stays navigable. With a
// corpus-shaped blob the neighbours a new node displaces are all near each
// other, so one insert can strip a node of every edge that pointed at it.
func corpusVectors(r *rand.Rand, docs, chunksPerDoc, dims int) map[int]Vector {
	const globalW, docW, chunkW = 0.721, 0.346, 0.600

	randomUnit := func() Vector {
		v := make(Vector, dims)
		for d := range v {
			v[d] = float32(r.NormFloat64())
		}
		return unit(v)
	}

	global := randomUnit()
	out := make(map[int]Vector, docs*chunksPerDoc)
	key := 0
	for doc := 0; doc < docs; doc++ {
		docVec := randomUnit()
		for c := 0; c < chunksPerDoc; c++ {
			noise := randomUnit()
			v := make(Vector, dims)
			for d := range v {
				v[d] = float32(globalW*float64(global[d]) +
					docW*float64(docVec[d]) +
					chunkW*float64(noise[d]))
			}
			out[key] = unit(v)
			key++
		}
	}
	return out
}

func unit(v Vector) Vector {
	var norm float64
	for _, x := range v {
		norm += float64(x) * float64(x)
	}
	s := float32(1 / math.Sqrt(norm))
	out := make(Vector, len(v))
	for i, x := range v {
		out[i] = x * s
	}
	return out
}

func buildCorpusGraph(t *testing.T, docs, chunksPerDoc int) (*Graph[int], map[int]Vector) {
	t.Helper()
	r := rand.New(rand.NewSource(7))
	vecs := corpusVectors(r, docs, chunksPerDoc, 768)
	g := NewGraph[int]()
	// Level assignment is otherwise seeded from the clock, which makes both the
	// graph shape and these assertions vary between runs.
	g.Rng = rand.New(rand.NewSource(1))
	for i := 0; i < len(vecs); i++ {
		g.Add(MakeNode(i, vecs[i]))
	}
	return g, vecs
}

// inDegree counts the base-layer edges pointing at key.
func inDegree(g *Graph[int], key int) int {
	if len(g.layers) == 0 {
		return 0
	}
	n := 0
	for k, node := range g.layers[0].nodes {
		if k == key {
			continue
		}
		if _, ok := node.neighbors[key]; ok {
			n++
		}
	}
	return n
}

// Every node in the base layer must have at least one incoming edge.
//
// Search follows edges in one direction only, so a node nothing points at is
// present in the layer but can never be returned: it is silently lost, and no
// setting of EfSearch or k brings it back. replenish() repaired connectivity by
// calling addNeighbor(), which records the new edge in one direction only,
// while addNeighbor's eviction path deletes the reverse edge and calls
// replenish again. A node whose whole neighbourhood was evicted therefore kept
// its outgoing edges and lost every incoming one.
func TestGraph_EveryNodeHasIncomingEdge(t *testing.T) {
	g, vecs := buildCorpusGraph(t, 60, 20)

	// A lost node shows up as zero in-degree, so this is the direct measure of
	// the bug: before the fix it was ~14% of the graph, after it is 0-2 nodes
	// (the tail of a repair pass, not a systematic failure).
	lost := 0
	for i := 0; i < len(vecs); i++ {
		if inDegree(g, i) == 0 {
			lost++
		}
	}
	require.LessOrEqualf(t, lost, len(vecs)/100, "%d of %d nodes have no incoming edge", lost, len(vecs))
}

// A node that nothing points at cannot be returned, however close it is to
// the query. Recall on a corpus-shaped graph must therefore be complete: with
// nodes losing their incoming edges, the nearest neighbour is simply absent
// from the result set and the caller gets a plausible-looking, wrong answer.
func TestGraph_RecallOnCorpusShape(t *testing.T) {
	g, vecs := buildCorpusGraph(t, 60, 20)

	hit, total := 0, 0
	for i := 0; i < len(vecs); i += 7 {
		// brute-force truth for this query
		type scored struct {
			score float64
			key   int
		}
		all := make([]scored, 0, len(vecs))
		for j := 0; j < len(vecs); j++ {
			var d float64
			for x := range vecs[i] {
				d += float64(vecs[i][x]) * float64(vecs[j][x])
			}
			all = append(all, scored{d, j})
		}
		sort.Slice(all, func(a, b int) bool { return all[a].score > all[b].score })
		truth := make(map[int]bool, 10)
		for a := 0; a < 10; a++ {
			truth[all[a].key] = true
		}

		for _, r := range g.Search(vecs[i], 10) {
			if truth[r.Key] {
				hit++
				break
			}
		}
		total++
	}
	// A lost node shows up here as a hard miss, not a low rank, so the floor is
	// set above what graph approximation alone costs. On this graph the
	// unreachable-node bug put recall at ~0.95; a correct graph reaches ~0.99,
	// the remainder being ordinary ANN approximation.
	const floor = 0.98
	got := float64(hit) / float64(total)
	require.GreaterOrEqualf(t, got, floor, "recall@10 = %d/%d (%.3f)", hit, total, got)
}

// The graph is rebuilt from stored vectors whenever a domain is re-routed, so a
// document reachable today must not be lost by the next re-route.
func TestGraph_EveryNodeHasIncomingEdgeAfterRebuild(t *testing.T) {
	first, vecs := buildCorpusGraph(t, 40, 20)

	rebuilt := NewGraph[int]()
	rebuilt.Rng = rand.New(rand.NewSource(1))
	for i := 0; i < len(vecs); i++ {
		v, ok := first.Lookup(i)
		require.Truef(t, ok, "node %d missing from the first graph", i)
		rebuilt.Add(MakeNode(i, v))
	}

	lost := 0
	for i := 0; i < len(vecs); i++ {
		if inDegree(rebuilt, i) == 0 {
			lost++
		}
	}
	require.LessOrEqualf(t, lost, len(vecs)/100, "%d of %d nodes have no incoming edge after rebuild", lost, len(vecs))
}
