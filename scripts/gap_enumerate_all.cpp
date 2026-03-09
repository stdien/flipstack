// Full pipeline: generate random permutations, solve optimally, enumerate
// all solution paths, and optionally build binary trie directly.
// Parallelized with OpenMP — one thread per permutation.
//
// Usage: gap_enumerate_all [options]
//   --n N              Permutation size (default: 20)
//   --num N            Number of random permutations (default: 10000)
//   --max-solutions N  Max solutions per perm before truncation (default: 10000)
//   --seed N           Random seed (default: 42)
//   --output PATH      Output JSON path (default: solutions.json)
//   --trie PATH        Build binary trie directly (skips JSON)
//   --csv PATH         Read perms from CSV instead of generating

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

static constexpr int MAXN = 128;
// Upper bound on pancake sorting distance: (5n+5)/3
static constexpr int MAX_DEPTH = (5 * MAXN + 5) / 3 + 1;

inline void flip(int8_t* p, int k) {
    int i = 0, j = k - 1;
    while (i < j) {
        int8_t tmp = p[i];
        p[i] = p[j];
        p[j] = tmp;
        i++; j--;
    }
}

inline int gap_h(const int8_t* p, int n) {
    int gaps = 0;
    for (int i = 0; i < n - 1; i++) {
        if (std::abs((int)p[i] - (int)p[i + 1]) != 1) gaps++;
    }
    if (std::abs((int)p[n - 1] - n) != 1) gaps++;
    return gaps;
}

inline bool is_sorted(const int8_t* p, int n) {
    for (int i = 0; i < n; i++) {
        if (p[i] != i) return false;
    }
    return true;
}

inline int flip_gap_delta(const int8_t* p, int k, int n) {
    int right = (k == n) ? n : (int)p[k];
    int old_gap = (std::abs((int)p[k - 1] - right) != 1) ? 1 : 0;
    int new_gap = (std::abs((int)p[0] - right) != 1) ? 1 : 0;
    return new_gap - old_gap;
}

// Phase 1: check if solution exists at given depth
bool dfs_exists(int8_t* work, int depth, int max_depth, int n,
                int gaps, int slack) {
    if (depth == max_depth)
        return is_sorted(work, n);
    if (gaps < 0 || gaps > max_depth - depth)
        return false;

    for (int k = 2; k <= n; k++) {
        int delta = flip_gap_delta(work, k, n);
        if (delta < 0) {
            flip(work, k);
            if (dfs_exists(work, depth + 1, max_depth, n,
                           gaps + delta, slack)) {
                flip(work, k);
                return true;
            }
            flip(work, k);
        } else if (slack > 0) {
            flip(work, k);
            if (dfs_exists(work, depth + 1, max_depth, n,
                           gaps + delta, slack - 1)) {
                flip(work, k);
                return true;
            }
            flip(work, k);
        }
    }
    return false;
}

// Phase 1.5: count solutions with early stop at limit
long long dfs_count(int8_t* work, int depth, int max_depth, int n,
                    int gaps, int slack, long long limit) {
    if (depth == max_depth)
        return is_sorted(work, n) ? 1 : 0;
    if (gaps < 0 || gaps > max_depth - depth)
        return 0;

    long long total = 0;
    for (int k = 2; k <= n; k++) {
        int delta = flip_gap_delta(work, k, n);
        if (delta < 0) {
            flip(work, k);
            total += dfs_count(work, depth + 1, max_depth, n,
                               gaps + delta, slack, limit);
            flip(work, k);
            if (total > limit) return total;
        } else if (slack > 0) {
            flip(work, k);
            total += dfs_count(work, depth + 1, max_depth, n,
                               gaps + delta, slack - 1, limit);
            flip(work, k);
            if (total > limit) return total;
        }
    }
    return total;
}

// Phase 2: enumerate all solution paths
void dfs_enumerate(int8_t* work, int depth, int max_depth, int n,
                   int gaps, int slack, int16_t* cur_path,
                   std::vector<std::vector<int16_t>>& solutions,
                   int max_solutions) {
    if ((int)solutions.size() >= max_solutions) return;
    if (depth == max_depth) {
        if (is_sorted(work, n)) {
            solutions.emplace_back(cur_path, cur_path + max_depth);
        }
        return;
    }
    if (gaps < 0 || gaps > max_depth - depth) return;

    for (int k = 2; k <= n; k++) {
        if ((int)solutions.size() >= max_solutions) return;
        int delta = flip_gap_delta(work, k, n);
        if (delta < 0) {
            cur_path[depth] = (int16_t)k;
            flip(work, k);
            dfs_enumerate(work, depth + 1, max_depth, n, gaps + delta, slack,
                          cur_path, solutions, max_solutions);
            flip(work, k);
        } else if (slack > 0) {
            cur_path[depth] = (int16_t)k;
            flip(work, k);
            dfs_enumerate(work, depth + 1, max_depth, n, gaps + delta,
                          slack - 1, cur_path, solutions, max_solutions);
            flip(work, k);
        }
    }
}

struct Result {
    std::vector<int8_t> perm;
    int sol_len;
    long long count;
    bool truncated;
    std::vector<std::vector<int16_t>> solutions;
};

// Generate unique random permutations using Fisher-Yates shuffle
std::vector<std::vector<int8_t>> generate_perms(int n, int num, uint64_t seed) {
    // Clamp to n! to avoid infinite loop
    long long fact = 1;
    for (int i = 2; i <= n && fact <= num; i++) fact *= i;
    if (num > (int)fact) {
        fprintf(stderr, "Warning: num=%d > %d! = %lld, clamping\n", num, n, fact);
        num = (int)fact;
    }

    std::mt19937_64 rng(seed);
    std::set<std::vector<int8_t>> seen;
    std::vector<std::vector<int8_t>> perms;
    perms.reserve(num);

    std::vector<int8_t> base(n);
    for (int i = 0; i < n; i++) base[i] = (int8_t)i;

    while ((int)perms.size() < num) {
        std::vector<int8_t> p = base;
        // Fisher-Yates shuffle
        for (int i = n - 1; i > 0; i--) {
            std::uniform_int_distribution<int> dist(0, i);
            int j = dist(rng);
            std::swap(p[i], p[j]);
        }
        if (seen.insert(p).second) {
            perms.push_back(std::move(p));
        }
    }
    return perms;
}

int parse_perm(const char* s, int8_t* out) {
    int n = 0;
    while (*s) {
        if (*s == '"') { s++; continue; }
        char* end;
        long val = strtol(s, &end, 10);
        if (end == s) { s++; continue; }  // skip non-numeric
        if (n >= MAXN) return n;  // prevent overflow
        out[n++] = (int8_t)val;
        s = end;
        if (*s == ',') s++;
    }
    return n;
}

std::vector<std::vector<int8_t>> load_csv(const char* path, int filter_n) {
    FILE* f = fopen(path, "r");
    if (!f) { perror("fopen"); exit(1); }

    std::vector<std::vector<int8_t>> perms;
    char line[8192];
    fgets(line, sizeof(line), f);  // skip header

    while (fgets(line, sizeof(line), f)) {
        char* p1 = strchr(line, ',');
        if (!p1) continue;
        *p1 = '\0';
        char* p2 = strchr(p1 + 1, ',');
        if (!p2) continue;
        *p2 = '\0';
        int n = atoi(p1 + 1);
        if (filter_n > 0 && n != filter_n) continue;
        char* q1 = strchr(p2 + 1, '"');
        if (!q1) continue;
        char* q2 = strchr(q1 + 1, '"');
        if (!q2) continue;
        *q2 = '\0';
        int8_t perm[MAXN];
        int pn = parse_perm(q1 + 1, perm);
        if (pn != n) continue;
        perms.emplace_back(perm, perm + pn);
    }
    fclose(f);
    return perms;
}

void write_json(const std::vector<Result>& results, int n,
                const char* output_path, int max_solutions) {
    FILE* f = fopen(output_path, "w");
    if (!f) { perror("fopen output"); exit(1); }

    int num_solved = 0, num_truncated = 0;
    for (auto& r : results) {
        if (r.sol_len >= 0) num_solved++;
        if (r.truncated) num_truncated++;
    }

    fprintf(f, "{\"n\":%d,\"max_slack\":-1,\"num_perms\":%d,"
               "\"num_solved\":%d,\"num_truncated\":%d,\"results\":[\n",
            n, (int)results.size(), num_solved, num_truncated);

    for (size_t i = 0; i < results.size(); i++) {
        auto& r = results[i];
        fprintf(f, "{\"perm\":[");
        for (size_t j = 0; j < r.perm.size(); j++) {
            if (j > 0) fputc(',', f);
            fprintf(f, "%d", (int)r.perm[j]);
        }
        fprintf(f, "],\"sol_len\":%d,\"count\":%lld,\"truncated\":%s",
                r.sol_len, r.count, r.truncated ? "true" : "false");

        if (!r.truncated && r.sol_len >= 0) {
            fprintf(f, ",\"solutions\":[");
            for (size_t si = 0; si < r.solutions.size(); si++) {
                if (si > 0) fputc(',', f);
                fputc('[', f);
                for (size_t mi = 0; mi < r.solutions[si].size(); mi++) {
                    if (mi > 0) fputc(',', f);
                    fprintf(f, "%d", (int)r.solutions[si][mi]);
                }
                fputc(']', f);
            }
            fputc(']', f);
        }

        fprintf(f, "}%s\n", (i + 1 < results.size()) ? "," : "");
    }

    fprintf(f, "]}\n");
    fclose(f);
}

// ========== Trie builder ==========

struct TrieNode {
    std::vector<int8_t> state;
    int16_t depth;
    std::map<int, int64_t> fwd_edges;  // move -> count (toward identity)
};

struct Trie {
    int n;
    std::vector<TrieNode> nodes;
    std::unordered_map<std::string, int> state_to_id;

    explicit Trie(int n_) : n(n_) {
        // Node 0 = identity
        TrieNode root;
        root.state.resize(n);
        for (int i = 0; i < n; i++) root.state[i] = (int8_t)i;
        root.depth = 0;
        nodes.push_back(std::move(root));
        state_to_id[std::string((char*)nodes[0].state.data(), n)] = 0;
    }

    int get_or_create(const int8_t* state, int16_t depth) {
        std::string key((const char*)state, n);
        auto it = state_to_id.find(key);
        if (it != state_to_id.end()) return it->second;
        int id = (int)nodes.size();
        TrieNode node;
        node.state.assign(state, state + n);
        node.depth = depth;
        nodes.push_back(std::move(node));
        state_to_id[key] = id;
        return id;
    }

    void ingest(const std::vector<int8_t>& /*perm*/,
                const std::vector<int16_t>& flips) {
        if (flips.empty()) return;

        // Reverse flip sequence: replay from identity
        std::vector<int8_t> state(n);
        for (int i = 0; i < n; i++) state[i] = (int8_t)i;
        int parent_id = 0;

        for (int step = (int)flips.size() - 1; step >= 0; step--) {
            int move = (int)flips[step];
            // Apply flip to state
            flip(state.data(), move);
            int16_t child_depth = (int16_t)((int)flips.size() - step);

            int child_id = get_or_create(state.data(), child_depth);

            // fwd: child -> parent (toward identity)
            nodes[child_id].fwd_edges[move] += 1;

            parent_id = child_id;
        }
    }
};

void write_trie(const Trie& trie, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) { perror("fopen trie"); exit(1); }

    int n = trie.n;
    int64_t num_nodes = (int64_t)trie.nodes.size();

    // Count edges
    int64_t num_fwd = 0;
    int16_t max_depth = 0;
    for (auto& node : trie.nodes) {
        num_fwd += (int64_t)node.fwd_edges.size();
        if (node.depth > max_depth) max_depth = node.depth;
    }

    // Write header (24 bytes): magic(4) + n(1) + max_depth(1) + pad(2) + num_nodes(8) + num_fwd(8)
    char header[24] = {};
    memcpy(header, "TRIE", 4);
    header[4] = (uint8_t)n;
    header[5] = (uint8_t)max_depth;
    // header[6..7] = pad (already zero)
    memcpy(header + 8, &num_nodes, 8);
    memcpy(header + 16, &num_fwd, 8);
    fwrite(header, 1, 24, f);

    // Nodes: num_nodes * n bytes (int8) — raw permutations
    for (auto& node : trie.nodes)
        fwrite(node.state.data(), 1, n, f);

    // Depths: num_nodes * 2 bytes (int16)
    for (auto& node : trie.nodes)
        fwrite(&node.depth, 2, 1, f);

    // Forward offsets: num_nodes * 4 bytes (int32)
    {
        int32_t off = 0;
        for (auto& node : trie.nodes) {
            fwrite(&off, 4, 1, f);
            off += (int32_t)node.fwd_edges.size();
        }
    }

    // Forward num: num_nodes * 1 byte (uint8)
    for (auto& node : trie.nodes) {
        uint8_t nc = (uint8_t)node.fwd_edges.size();
        fwrite(&nc, 1, 1, f);
    }

    // Forward moves: num_fwd * 1 byte (uint8)
    for (auto& node : trie.nodes) {
        for (auto& [move, count] : node.fwd_edges) {
            uint8_t m = (uint8_t)move;
            fwrite(&m, 1, 1, f);
        }
    }

    // Forward counts: num_fwd * 4 bytes (int32)
    for (auto& node : trie.nodes) {
        for (auto& [move, count] : node.fwd_edges) {
            int32_t c = (int32_t)count;
            fwrite(&c, 4, 1, f);
        }
    }

    fclose(f);

    int64_t file_size = 48 + (int64_t)num_nodes * n + num_nodes * 2
                      + num_nodes * 4 + num_nodes + num_fwd + num_fwd * 4;
    fprintf(stderr, "Trie v2: %ld nodes, %ld fwd edges, "
            "max_depth=%d (%ld bytes)\n",
            (long)num_nodes, (long)num_fwd, (int)max_depth, (long)file_size);
}

// ========== Main ==========

int main(int argc, char** argv) {
    int n = 20;
    int num = 10000;
    int max_solutions = 10000;
    uint64_t seed = 42;
    const char* output_path = "solutions.json";
    const char* trie_path = nullptr;
    const char* csv_path = nullptr;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--n") == 0 && i + 1 < argc)
            n = atoi(argv[++i]);
        else if (strcmp(argv[i], "--num") == 0 && i + 1 < argc)
            num = atoi(argv[++i]);
        else if (strcmp(argv[i], "--max-solutions") == 0 && i + 1 < argc)
            max_solutions = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
            seed = (uint64_t)atoll(argv[++i]);
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)
            output_path = argv[++i];
        else if (strcmp(argv[i], "--trie") == 0 && i + 1 < argc)
            trie_path = argv[++i];
        else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc)
            csv_path = argv[++i];
        else {
            fprintf(stderr, "Usage: %s [--n N] [--num N] [--max-solutions N] "
                    "[--seed N] [--output PATH] [--trie PATH] [--csv PATH]\n", argv[0]);
            return 1;
        }
    }

    if (n < 1 || n > MAXN) {
        fprintf(stderr, "Error: n must be between 1 and %d\n", MAXN);
        return 1;
    }
    if (trie_path && n > 127) {
        fprintf(stderr, "Error: --trie requires n <= 127\n");
        return 1;
    }

    // Load or generate permutations
    std::vector<std::vector<int8_t>> perms;
    if (csv_path) {
        perms = load_csv(csv_path, n);
        num = (int)perms.size();
        fprintf(stderr, "Loaded %d permutations from %s\n", num, csv_path);
    } else {
        fprintf(stderr, "Generating %d random permutations of size %d (seed=%lu)...\n",
                num, n, (unsigned long)seed);
        perms = generate_perms(n, num, seed);
        num = (int)perms.size();  // may have been clamped to n!
    }

    fprintf(stderr, "Solving %d permutations (n=%d, max_solutions=%d)\n",
            num, n, max_solutions);

    std::vector<Result> results(num);
    std::atomic<int> progress{0};

    double t0 = 0;
    #ifdef _OPENMP
    t0 = omp_get_wtime();
    #endif

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num; i++) {
        Result& r = results[i];
        r.perm = perms[i];
        r.sol_len = -1;
        r.count = 0;
        r.truncated = false;

        int pn = (int)perms[i].size();
        int gh = gap_h(perms[i].data(), pn);

        if (gh == 0) {
            r.sol_len = 0;
            r.count = 1;
            r.solutions.emplace_back();  // empty flip sequence
            ++progress;
            continue;
        }

        int8_t work[MAXN];

        // Phase 1: find optimal length (unlimited slack)
        for (int slack = 0; ; slack++) {
            memcpy(work, perms[i].data(), pn);
            if (dfs_exists(work, 0, gh + slack, pn, gh, slack)) {
                r.sol_len = gh + slack;
                break;
            }
        }

        int slack = r.sol_len - gh;

        // Phase 1.5: count with early stop
        memcpy(work, perms[i].data(), pn);
        long long count = dfs_count(work, 0, r.sol_len, pn, gh, slack,
                                    (long long)max_solutions + 1);

        if (count > max_solutions) {
            r.count = -1;
            r.truncated = true;
        } else {
            r.count = count;

            // Phase 2: enumerate all solutions
            int16_t cur_path[MAX_DEPTH];
            memcpy(work, perms[i].data(), pn);
            r.solutions.reserve((int)count);
            dfs_enumerate(work, 0, r.sol_len, pn, gh, slack,
                          cur_path, r.solutions, (int)count);
        }

        int done = ++progress;
        if (done % 100 == 0) {
            #ifdef _OPENMP
            fprintf(stderr, "Progress: %d/%d (%.1fs)\n", done, num,
                    omp_get_wtime() - t0);
            #else
            fprintf(stderr, "Progress: %d/%d\n", done, num);
            #endif
        }
    }

    #ifdef _OPENMP
    double elapsed = omp_get_wtime() - t0;
    #else
    double elapsed = 0;
    #endif

    int num_solved = 0, num_truncated = 0;
    long long total_solutions = 0;
    for (auto& r : results) {
        if (r.sol_len >= 0) num_solved++;
        if (r.truncated) num_truncated++;
        if (!r.truncated) total_solutions += (long long)r.solutions.size();
    }

    fprintf(stderr, "Done: %d/%d solved, %d truncated, %lld total solutions "
            "(%.1fs)\n", num_solved, num, num_truncated, total_solutions,
            elapsed);

    if (trie_path) {
        fprintf(stderr, "Building trie...\n");
        Trie trie(n);
        long long ingested = 0;
        for (auto& r : results) {
            if (r.sol_len < 0 || r.truncated) continue;
            for (auto& sol : r.solutions) {
                trie.ingest(r.perm, sol);
                ingested++;
            }
        }
        fprintf(stderr, "Ingested %lld solution paths into %d nodes\n",
                ingested, (int)trie.nodes.size());
        write_trie(trie, trie_path);
    } else {
        fprintf(stderr, "Writing JSON to %s...\n", output_path);
        write_json(results, n, output_path, max_solutions);
    }

    fprintf(stderr, "Done.\n");
    return 0;
}
