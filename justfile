# Run criterion
json:
    cargo criterion --bench bench --message-format=json --plotting-backend=disabled -- --quiet --quick | \
    jq -s '[.[] | select(.reason == "benchmark-complete") \
    | { \
        id: .id, \
        mean: .mean.estimate, \
      }]' > results/bench.json

build:
    cargo build -r --bench bench

bench bench='' *args='':
    cargo bench --bench bench -- --quiet -n "{{bench}}" {{args}}

# record time usage
record bench='' *args='': build
    perf record cargo bench --bench bench -- --profile-time 2 "{{bench}}" {{args}}
    perf report -n
report:
    perf report -n

cpufreq:
    sudo cpupower frequency-set --governor performance -d 2.6GHz -u 2.6GHz
powersave:
    sudo cpupower frequency-set --governor powersave -d 1.0GHz -u 2.6GHz

allow-profiling:
    sudo sh -c 'echo 1 >/proc/sys/kernel/perf_event_paranoid'
