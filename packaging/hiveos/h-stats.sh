#!/usr/bin/env bash
# The HiveOS agent sources this: it must set $khs (total kH/s) and $stats (JSON).
# Every GPU process serves its own /hiveos on a port listed in CUSTOM_PORTS_FILE; temperatures
# and fans come from the agent's $gpu_stats, matched by PCI bus.

cppminer_stats() {
    local dir="${MINER_DIR:-/hive/miners/custom}/${CUSTOM_MINER:-cppminer}"
    . "$dir/h-manifest.conf"
    local ports="$CUSTOM_API_PORT" port j all="[]"
    [[ -s $CUSTOM_PORTS_FILE ]] && ports=$(cat "$CUSTOM_PORTS_FILE")
    for port in $ports; do
        j=$(curl -s --max-time 2 "http://127.0.0.1:$port/hiveos")
        [[ -z $j ]] && continue
        all=$(jq -c --argjson x "$j" '. + [$x]' <<< "$all" 2>/dev/null) || all="[]"
    done

    local g="${gpu_stats:-}"
    [[ -z $g ]] && g="{}"
    local out
    out=$(jq -c --argjson g "$g" --arg ver "$CUSTOM_VERSION" '
        def hex2dec: ascii_downcase | explode
            | reduce .[] as $c (0; . * 16 + (if $c >= 97 then $c - 87 else $c - 48 end));
        ([.[].stats.bus_numbers // [] | .[]]) as $bus
        | (($g.busids // []) | map(tostring | split(":") | if length >= 2 then .[-2] else "zz" end
                                   | if test("^[0-9a-fA-F]+$") then hex2dec else -1 end)) as $gb
        | {
            khs: ([.[].khs] | add // 0),
            stats: {
                hs: ([.[].stats.hs // [] | .[]]),
                hs_units: "khs",
                temp: [$bus[] as $b | ($gb | index($b)) as $i
                       | if $i == null then 0 else ($g.temp[$i] // 0) end],
                fan: [$bus[] as $b | ($gb | index($b)) as $i
                      | if $i == null then 0 else ($g.fan[$i] // 0) end],
                uptime: ([.[].stats.uptime] | max // 0),
                ver: (.[0].stats.ver // $ver),
                ar: [([.[].stats.ar[0]] | add // 0), ([.[].stats.ar[1]] | add // 0)],
                algo: (.[0].stats.algo // ""),
                bus_numbers: $bus
            }
        }' <<< "$all" 2>/dev/null)
    if [[ -z $out ]]; then
        khs=0
        stats='{"hs":[],"hs_units":"khs","temp":[],"fan":[],"uptime":0,"ar":[0,0],"bus_numbers":[]}'
        return
    fi
    khs=$(jq -r '.khs' <<< "$out")
    stats=$(jq -c '.stats' <<< "$out")
}

cppminer_stats
