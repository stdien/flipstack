// Gap DFS solver with configurable slack for pancake sorting.
// At each step: if slack remaining > 0, try all moves (costs 1 slack per non-reducing).
//               if slack remaining == 0, only gap-reducing moves.
// Parallelized with OpenMP — one thread per permutation.
//
// Usage: gap_slack_dfs <test.csv> [max_n=12] [max_slack=2]

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

static constexpr int MAXN = 128;

inline void flip(int8_t* p, int k) {
    int i = 0, j = k - 1;
    while (i < j) {
        int8_t tmp = p[i];
        p[i] = p[j];
        p[j] = tmp;
        i++; j--;
    }
}

inline int gap_h_val(const int8_t* p, int n) {
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

// Compute gap delta for flip(k): O(1)
inline int flip_gap_delta(const int8_t* p, int k, int n) {
    int right = (k == n) ? n : (int)p[k];
    int old_left = (int)p[k - 1];
    int new_left = (int)p[0];
    int old_gap = (std::abs(old_left - right) != 1) ? 1 : 0;
    int new_gap = (std::abs(new_left - right) != 1) ? 1 : 0;
    return new_gap - old_gap;
}

struct Task {
    int id, n;
    int8_t perm[MAXN];
};

struct Result {
    int id, n, gh, sol_len;
    int flips[MAXN * 2];
};

void dfs(int8_t* work, int* cur_flips, int* best_flips, int& best_len,
         int depth, int max_depth, int n, int gaps_remaining, int slack_remaining) {
    if (depth == max_depth) {
        if (is_sorted(work, n)) {
            if (best_len < 0 || depth < best_len) {
                best_len = depth;
                memcpy(best_flips, cur_flips, depth * sizeof(int));
            }
        }
        return;
    }
    if (gaps_remaining < 0) return;
    int remaining = max_depth - depth;
    if (gaps_remaining > remaining) return;

    for (int k = 2; k <= n; k++) {
        int delta = flip_gap_delta(work, k, n);
        if (delta < 0) {
            // Gap-reducing: always try
            cur_flips[depth] = k;
            flip(work, k);
            dfs(work, cur_flips, best_flips, best_len,
                depth + 1, max_depth, n, gaps_remaining + delta, slack_remaining);
            flip(work, k);
        } else if (slack_remaining > 0) {
            // Gap-neutral or gap-increasing: costs 1 slack
            cur_flips[depth] = k;
            flip(work, k);
            dfs(work, cur_flips, best_flips, best_len,
                depth + 1, max_depth, n, gaps_remaining + delta, slack_remaining - 1);
            flip(work, k);
        }
    }
}

int parse_perm(const char* s, int8_t* out) {
    int n = 0;
    while (*s) {
        if (*s == '"') { s++; continue; }
        char* end;
        long val = strtol(s, &end, 10);
        out[n++] = (int8_t)val;
        s = end;
        if (*s == ',') s++;
    }
    return n;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <test.csv> [max_n=12] [max_slack=2]\n", argv[0]);
        return 1;
    }
    int max_n = 12;
    if (argc >= 3) max_n = atoi(argv[2]);
    int max_slack = 2;
    if (argc >= 4) max_slack = atoi(argv[3]);
    if (max_slack > MAXN) max_slack = MAXN;  // clamp to buffer capacity

    FILE* f = fopen(argv[1], "r");
    if (!f) { perror("fopen"); return 1; }

    std::vector<Task> tasks;
    char line[8192];
    fgets(line, sizeof(line), f);

    while (fgets(line, sizeof(line), f)) {
        char* p1 = strchr(line, ',');
        if (!p1) continue;
        *p1 = '\0';
        int id = atoi(line);
        char* p2 = strchr(p1 + 1, ',');
        if (!p2) continue;
        *p2 = '\0';
        int n = atoi(p1 + 1);
        if (n > max_n) continue;
        char* q1 = strchr(p2 + 1, '"');
        if (!q1) continue;
        char* q2 = strchr(q1 + 1, '"');
        if (!q2) continue;
        *q2 = '\0';
        Task t;
        t.id = id;
        t.n = n;
        int pn = parse_perm(q1 + 1, t.perm);
        if (pn != n) continue;
        tasks.push_back(t);
    }
    fclose(f);
    fprintf(stderr, "Loaded %zu tasks (max_n=%d, max_slack=%d)\n", tasks.size(), max_n, max_slack);

    std::vector<Result> results(tasks.size());
    std::atomic<int> progress{0};

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < tasks.size(); i++) {
        const Task& t = tasks[i];
        Result& r = results[i];
        r.id = t.id;
        r.n = t.n;
        r.gh = gap_h_val(t.perm, t.n);
        r.sol_len = -1;

        int8_t work[MAXN];
        int cur_flips[MAXN * 2];

        // Try increasing slack until solution found
        for (int slack = 0; slack <= max_slack && r.sol_len < 0; slack++) {
            memcpy(work, t.perm, t.n);
            dfs(work, cur_flips, r.flips, r.sol_len, 0, r.gh + slack, t.n, r.gh, slack);
        }

        int done = ++progress;
        if (done % 100 == 0) {
            fprintf(stderr, "Processed %d/%zu\n", done, tasks.size());
        }
    }

    printf("id,n,gap_h,sol_len\n");
    int solved_exact = 0, solved_slack = 0, unsolved = 0;
    for (auto& r : results) {
        printf("%d,%d,%d,%d", r.id, r.n, r.gh, r.sol_len);
        if (r.sol_len > 0) {
            printf(",");
            for (int j = 0; j < r.sol_len; j++) {
                if (j > 0) printf(".");
                printf("R%d", r.flips[j]);
            }
            if (r.sol_len == r.gh) solved_exact++;
            else solved_slack++;
        } else {
            unsolved++;
        }
        printf("\n");
    }
    fprintf(stderr, "Done: exact=%d, slack=%d, unsolved=%d\n", solved_exact, solved_slack, unsolved);
    return 0;
}
