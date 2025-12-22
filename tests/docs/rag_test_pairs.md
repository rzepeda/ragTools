# RAG Testing: 15 Text-Query Pairs

## Purpose
These pairs test different retrieval capabilities: semantic similarity, keyword matching, hierarchical understanding, multi-hop reasoning, negation handling, numerical precision, and edge cases.

---

## Pair 1: Semantic Similarity (Basic)
**Tests:** Vector embeddings, semantic understanding

**Text:**
The Arctic fox, also known as the white fox or polar fox, is remarkably adapted to survive in extreme cold. Its thick fur provides insulation with the warmest coat of any mammal, while its compact body shape minimizes heat loss. During winter, its coat turns white for camouflage against snow, and in summer it changes to brown or gray to blend with tundra vegetation. These foxes can survive temperatures as low as -70°C.

**Query:** What adaptations help polar foxes survive freezing temperatures?

**Expected:** Should retrieve based on semantic understanding of "polar foxes" = "Arctic fox" and "freezing temperatures" = "extreme cold"

---

## Pair 2: Keyword Precision (Technical)
**Tests:** BM25, exact keyword matching

**Text:**
The TCP three-way handshake establishes a connection through SYN, SYN-ACK, and ACK packets. First, the client sends a SYN packet with a random sequence number. The server responds with SYN-ACK, acknowledging the client's sequence number and providing its own. Finally, the client sends an ACK packet, completing the handshake. This mechanism ensures both parties are ready for data transmission and prevents half-open connections.

**Query:** Explain the TCP three-way handshake sequence

**Expected:** Should match exact technical terms: "TCP", "three-way handshake", "SYN", "SYN-ACK", "ACK"

---

## Pair 3: Hierarchical Context
**Tests:** Parent-child chunk relationships, context expansion

**Text:**
Renaissance art underwent three distinct phases. The Early Renaissance (1400-1475) focused on perspective and anatomy, with artists like Masaccio pioneering linear perspective. The High Renaissance (1475-1525) achieved perfect balance, exemplified by Leonardo da Vinci's sfumato technique and Michelangelo's sculptures. The Late Renaissance (1525-1600) introduced Mannerism, characterized by elongated figures and complex compositions, as seen in works by Parmigianino.

**Query:** What technique did Leonardo da Vinci develop during the High Renaissance?

**Expected:** Should retrieve the specific section about High Renaissance while understanding the broader context of Renaissance phases

---

## Pair 4: Multi-Hop Reasoning
**Tests:** Agentic RAG, graph traversal, relationship understanding

**Text:**
Marie Curie won her first Nobel Prize in Physics in 1903 alongside her husband Pierre Curie and Henri Becquerel for their work on radioactivity. After Pierre's death in 1906, she continued research and won her second Nobel Prize in Chemistry in 1911 for discovering radium and polonium. Her daughter, Irène Joliot-Curie, also won a Nobel Prize in Chemistry in 1935 for synthesizing new radioactive elements.

**Query:** How many Nobel Prizes did the Curie family win in total?

**Expected:** Requires connecting: Marie's 2 prizes + Irène's 1 prize = 3 total (multi-hop reasoning)

---

## Pair 5: Negation Handling
**Tests:** Query understanding, semantic precision

**Text:**
While most reptiles are cold-blooded ectotherms, the leatherback sea turtle exhibits regional endothermy. Unlike typical reptiles, it can maintain body temperatures up to 18°C warmer than surrounding water through metabolic heat generation and gigantothermy. This unique adaptation allows leatherbacks to hunt in frigid Arctic waters where other sea turtles cannot survive, pursuing jellyfish at depths exceeding 1,000 meters.

**Query:** What reptiles can regulate their body temperature unlike most reptiles?

**Expected:** Should understand "unlike most reptiles" and identify leatherback sea turtle as the exception

---

## Pair 6: Numerical Precision
**Tests:** Exact value retrieval, contextual understanding

**Text:**
The human brain contains approximately 86 billion neurons, each forming between 1,000 and 10,000 synapses with other neurons. This creates a network of roughly 100 trillion connections. The cerebral cortex, representing about 80% of the brain's mass, handles higher cognitive functions. Information travels through neurons at speeds ranging from 0.5 to 120 meters per second, depending on whether the axon is myelinated.

**Query:** How many neurons are in the human brain?

**Expected:** Should return the specific number "86 billion" with high precision

---

## Pair 7: Temporal Ordering
**Tests:** Understanding sequence, chronological relationships

**Text:**
The Apollo 11 mission achieved humanity's first lunar landing through carefully orchestrated stages. On July 16, 1969, the Saturn V rocket launched from Kennedy Space Center. Three days later, on July 19, the spacecraft entered lunar orbit. July 20 marked the historic moment when the Eagle lunar module landed in the Sea of Tranquility at 20:17 UTC. Neil Armstrong stepped onto the lunar surface six hours later at 02:56 UTC on July 21, followed by Buzz Aldrin nineteen minutes afterward.

**Query:** What happened between the lunar orbit insertion and the first steps on the moon?

**Expected:** Should retrieve the landing event (July 20) that occurred chronologically between orbit (July 19) and moonwalk (July 21)

---

## Pair 8: Comparative Analysis
**Tests:** Multi-query, contextual retrieval, comparison understanding

**Text:**
Photosynthesis and cellular respiration are complementary processes. Photosynthesis converts carbon dioxide and water into glucose and oxygen using light energy, occurring in chloroplasts of plant cells. The chemical equation is 6CO₂ + 6H₂O + light energy → C₆H₁₂O₆ + 6O₂. Cellular respiration reverses this process, breaking down glucose with oxygen to produce ATP, carbon dioxide, and water in mitochondria. All living cells perform cellular respiration, while only plant cells, algae, and some bacteria perform photosynthesis.

**Query:** What are the differences between photosynthesis and cellular respiration?

**Expected:** Should identify contrasts: location (chloroplasts vs mitochondria), direction (building vs breaking down glucose), organisms (plants only vs all cells)

---

## Pair 9: Acronym and Abbreviation
**Tests:** Entity recognition, term expansion

**Text:**
The United States Federal Bureau of Investigation (FBI) was established in 1908 as the Bureau of Investigation under the Department of Justice. It gained its current name in 1935. The FBI's primary mission includes protecting the United States from terrorist attacks, foreign intelligence operations, and cyber threats. Headquartered in Washington, D.C., the bureau operates 56 field offices across the country and maintains legal attaché offices in U.S. embassies worldwide.

**Query:** When was the FBI founded and what does it do?

**Expected:** Should match "FBI" with "Federal Bureau of Investigation" and extract founding year (1908) plus primary missions

---

## Pair 10: Causal Relationships
**Tests:** Self-reflective RAG, reasoning about cause and effect

**Text:**
The Irish Potato Famine of 1845-1852 was triggered by Phytophthora infestans, a water mold that devastated potato crops. Since potatoes comprised 80% of the Irish diet, the blight caused mass starvation. Approximately one million people died, and another million emigrated, reducing Ireland's population by 25%. British government policies exacerbated the crisis by continuing grain exports from Ireland and providing inadequate relief. This catastrophe fundamentally altered Ireland's demographics, culture, and relationship with Britain.

**Query:** Why did the potato blight have such devastating effects in Ireland?

**Expected:** Should identify multiple causal factors: potato dependence (80% of diet), inadequate government response, continued exports

---

## Pair 11: Definition with Context
**Tests:** Contextual retrieval, LLM-enriched chunks

**Text:**
Quantum entanglement occurs when particles become correlated such that the quantum state of one particle cannot be described independently of others, even when separated by vast distances. Einstein famously called this "spooky action at a distance." When measuring one entangled particle's property (like spin), the other particle's corresponding property is instantly determined, regardless of separation. This phenomenon doesn't violate relativity because no information is actually transmitted faster than light—the correlation exists, but random measurement outcomes prevent faster-than-light communication.

**Query:** What is quantum entanglement?

**Expected:** Should provide the definition while maintaining important contextual details (Einstein's characterization, no FTL communication)

---

## Pair 12: Ambiguous Reference Resolution
**Tests:** Coreference resolution, entity tracking

**Text:**
Amazon was founded by Jeff Bezos in 1994 as an online bookstore operating from his garage in Seattle. The company chose its name to convey scale, referencing the Amazon River, the world's largest river by volume. Initially focusing solely on books, it expanded into CDs, electronics, and eventually became the world's largest online retailer. Today, the corporation dominates cloud computing through AWS, produces consumer electronics like the Kindle, and operates streaming services, far exceeding its original scope.

**Query:** Why did the company choose that name?

**Expected:** Should understand "the company" refers to Amazon and "that name" refers to Amazon, retrieving the river reference explanation

---

## Pair 13: Procedural Knowledge
**Tests:** Step-by-step retrieval, ordered information

**Text:**
Performing the Heimlich maneuver on a choking adult requires specific steps. First, stand behind the person and wrap your arms around their waist. Make a fist with one hand and place it slightly above the navel, thumb side in. Grasp your fist with your other hand and press hard into the abdomen with a quick, upward thrust. Repeat this motion five times. If the object doesn't dislodge, check the person's mouth for the obstruction. Continue cycles of five abdominal thrusts until the object is expelled or emergency services arrive.

**Query:** How do you perform the Heimlich maneuver?

**Expected:** Should retrieve the complete procedure in correct sequential order

---

## Pair 14: Statistical Information
**Tests:** Late chunking, maintaining numerical context

**Text:**
Climate change impacts vary significantly by region. Arctic temperatures are rising twice as fast as the global average, with sea ice declining at 13% per decade since 1979. Mediterranean regions face 40% reduction in summer rainfall by 2100 under current emissions scenarios. Small island nations like Tuvalu and Maldives, with maximum elevations under 5 meters, face existential threats from projected sea level rise of 0.5 to 1.0 meters by 2100. Meanwhile, sub-Saharan Africa experiences increased drought frequency, with some regions seeing 30% decline in agricultural productivity.

**Query:** Which regions are most affected by climate change and by how much?

**Expected:** Should retrieve multiple statistical comparisons while maintaining their regional context

---

## Pair 15: Edge Case - Very Recent Information
**Tests:** Fine-tuned embeddings, specialized domain knowledge

**Text:**
The James Webb Space Telescope (JWST), launched on December 25, 2021, revolutionized astronomy by observing the universe in infrared wavelengths. Its 6.5-meter primary mirror, composed of 18 gold-plated beryllium segments, collects light from objects 13.5 billion years old. Positioned at the L2 Lagrange point, 1.5 million kilometers from Earth, JWST operates at -233°C to detect faint infrared signals. Early discoveries included detailed images of exoplanet atmospheres, ancient galaxies forming 300 million years after the Big Bang, and complex organic molecules in star-forming regions.

**Query:** What has the James Webb telescope discovered?

**Expected:** Should retrieve specific discoveries: exoplanet atmospheres, ancient galaxies (300M years post-Big Bang), organic molecules

---

## Testing Strategy Matrix

| Pair | Semantic | Keyword | Hierarchical | Multi-hop | Reranking | Graph | Contextual |
|------|----------|---------|--------------|-----------|-----------|-------|------------|
| 1    | âœ…       | -       | -            | -         | âœ…        | -     | -          |
| 2    | -        | âœ…      | -            | -         | âœ…        | -     | -          |
| 3    | âœ…       | -       | âœ…           | -         | âœ…        | -     | âœ…         |
| 4    | âœ…       | -       | -            | âœ…        | âœ…        | âœ…    | -          |
| 5    | âœ…       | -       | -            | -         | âœ…        | -     | âœ…         |
| 6    | âœ…       | âœ…      | -            | -         | âœ…        | -     | -          |
| 7    | âœ…       | -       | âœ…           | âœ…        | âœ…        | âœ…    | -          |
| 8    | âœ…       | -       | -            | âœ…        | âœ…        | -     | âœ…         |
| 9    | âœ…       | âœ…      | -            | -         | âœ…        | âœ…    | -          |
| 10   | âœ…       | -       | -            | âœ…        | âœ…        | âœ…    | âœ…         |
| 11   | âœ…       | âœ…      | -            | -         | âœ…        | -     | âœ…         |
| 12   | âœ…       | -       | -            | âœ…        | âœ…        | âœ…    | âœ…         |
| 13   | âœ…       | âœ…      | âœ…           | -         | âœ…        | -     | -          |
| 14   | âœ…       | âœ…      | -            | âœ…        | âœ…        | -     | âœ…         |
| 15   | âœ…       | âœ…      | -            | -         | âœ…        | -     | âœ…         |

## Evaluation Metrics

For each pair, measure:
1. **Retrieval Accuracy**: Did the correct text chunk get retrieved in top-k?
2. **Ranking Quality**: What rank was the correct chunk?
3. **Semantic Precision**: How well did the query match semantically?
4. **Keyword Overlap**: How many exact keyword matches?
5. **Latency**: Time to retrieve (ms)
6. **Cost**: API calls required (for LLM-based strategies)

## Expected Strategy Performance

**Best for each pair:**
- Pair 1: Semantic (vector embeddings)
- Pair 2: Keyword (BM25)
- Pair 3: Hierarchical RAG
- Pair 4: Agentic RAG or Knowledge Graph
- Pair 5: Semantic with Query Expansion
- Pair 6: Semantic or Keyword
- Pair 7: Hierarchical RAG
- Pair 8: Multi-Query RAG
- Pair 9: Keyword + Entity Recognition
- Pair 10: Self-Reflective RAG or Graph
- Pair 11: Contextual Retrieval
- Pair 12: Semantic with Reranking
- Pair 13: Hierarchical RAG
- Pair 14: Late Chunking or Contextual
- Pair 15: Fine-tuned Embeddings + Semantic

## Usage

```python
# Load all pairs
test_pairs = load_test_pairs("rag_test_pairs.md")

# Index all texts
for pair in test_pairs:
    indexer.index(pair.text, metadata={"pair_id": pair.id})

# Test each strategy
for strategy in strategies:
    for pair in test_pairs:
        results = strategy.retrieve(pair.query, top_k=5)
        evaluate(results, pair.text, pair.expected_behaviors)
```
