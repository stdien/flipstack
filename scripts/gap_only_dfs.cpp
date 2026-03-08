// Gap-only DFS solver for pancake sorting.
// Only explores gap-reducing moves (branching factor <= 2).
// Finds shortest solution of length exactly gap_h, or reports none exists.
// Parallelized with OpenMP — one thread per permutation.
//
// Usage: gap_only_dfs <test.csv> [max_n]
// Output: CSV lines: id,n,gap_h,gap_only_len,flips (or -1 if no solution)

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

inline int find_gap_reducing(const int8_t* p, int n, int moves[2]) {
    int v = p[0];
    int count = 0;
    for (int delta = -1; delta <= 1; delta += 2) {
        int target = v + delta;
        int k;
        if (target < 0) continue;
        if (target == n) {
            k = n;
        } else if (target > n) {
            continue;
        } else {
            k = -1;
            for (int i = 1; i < n; i++) {
                if (p[i] == target) { k = i; break; }
            }
            if (k <= 0) continue;
        }
        int right = (k == n) ? n : (int)p[k];
        int left = (int)p[k - 1];
        if (std::abs(left - right) != 1) {
            moves[count++] = k;
        }
    }
    return count;
}

struct Task {
    int id;
    int n;
    int8_t perm[MAXN];
};

struct Result {
    int id;
    int n;
    int gh;
    int sol_len;  // -1 if none
    int flips[MAXN];
};

void dfs(int8_t* work, int* current_flips, int* best_flips, int& best_len,
         int depth, int n, int target_gap) {
    if (target_gap == 0) {
        if (is_sorted(work, n)) {
            if (best_len < 0 || depth < best_len) {
                best_len = depth;
                memcpy(best_flips, current_flips, depth * sizeof(int));
            }
        }
        return;
    }

    int moves[2];
    int nmoves = find_gap_reducing(work, n, moves);
    for (int i = 0; i < nmoves; i++) {
        current_flips[depth] = moves[i];
        flip(work, moves[i]);
        dfs(work, current_flips, best_flips, best_len, depth + 1, n, target_gap - 1);
        flip(work, moves[i]);
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
        fprintf(stderr, "Usage: %s <test.csv> [max_n=50]\n", argv[0]);
        return 1;
    }
    int max_n = 50;
    if (argc >= 3) max_n = atoi(argv[2]);

    FILE* f = fopen(argv[1], "r");
    if (!f) { perror("fopen"); return 1; }

    // Read all tasks
    std::vector<Task> tasks;
    char line[8192];
    fgets(line, sizeof(line), f);  // skip header

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
        if (pn != n) {
            fprintf(stderr, "Warning: id=%d expected n=%d got %d elements\n", id, n, pn);
            continue;
        }
        tasks.push_back(t);
    }
    fclose(f);

    fprintf(stderr, "Loaded %zu tasks (max_n=%d)\n", tasks.size(), max_n);

    // Solve in parallel
    std::vector<Result> results(tasks.size());
    std::atomic<int> progress{0};

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < tasks.size(); i++) {
        const Task& t = tasks[i];
        Result& r = results[i];
        r.id = t.id;
        r.n = t.n;
        r.gh = gap_h(t.perm, t.n);
        r.sol_len = -1;

        int8_t work[MAXN];
        int current_flips[MAXN];
        memcpy(work, t.perm, t.n);

        dfs(work, current_flips, r.flips, r.sol_len, 0, t.n, r.gh);

        int done = ++progress;
        if (done % 200 == 0) {
            fprintf(stderr, "Processed %d/%zu perms\n", done, tasks.size());
        }
    }

    // Output results in order
    printf("id,n,gap_h,gap_only_len\n");
    int total_solved = 0;
    for (size_t i = 0; i < results.size(); i++) {
        const Result& r = results[i];
        printf("%d,%d,%d,%d", r.id, r.n, r.gh, r.sol_len);
        if (r.sol_len > 0) {
            printf(",");
            for (int j = 0; j < r.sol_len; j++) {
                if (j > 0) printf(".");
                printf("R%d", r.flips[j]);
            }
            total_solved++;
        }
        printf("\n");
    }

    fprintf(stderr, "Done: %d/%zu solved at gap_h\n", total_solved, results.size());
    return 0;
}
