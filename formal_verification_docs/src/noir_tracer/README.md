# Noir Tracer

A useful tool when debugging Noir programs is one that can generate an
execution trace of the program for the given inputs. Such a trace can be loaded
in a specialized program that visualizes the execution timeline and gives
insights into the behavior of the program during execution.

At [Blocksense](blocksense.network), we have built such a tool and integrated
it in the `nargo` command line interface. Note that the project is still an
early-alpha prototype, so there is still much to improve. Yet, it is already
powerful enough to handle a broad range of Noir programs.

We are using the [runtime
tracing](https://github.com/metacraft-labs/runtime_tracing) Rust library for
the tracing format.

## Installation

As already mentioned, the tool is integrated in the `nargo` CLI, but you need
to fetch our fork of the Noir programming language in order to get it. Follow
the [installation
guide](https://noir.blocksense.network/noir_plonky2_backend/getting_started#installation)
for the Noir PLONKY2 backend, as it describes how to fetch the fork and build
the system. Once you have `nargo` set up, you can use it in the following way.

## Using

In order to generate a trace for a Noir program, you need to navigate to its
root directory (the directory containing `Nargo.toml`). You also need to
specify the directory where the trace should be generated.

You could try one of the programs included in `test_programs`. Alternatively,
you can create a new one in the usual way:

```
nargo new simple_loop
cd simple_loop
```

Once that's done, write some text for the program (in `src/main.nr`):

```Rust
fn main(x: pub u64, y: u64) {
    let mut z: u64 = y;
    for _ in 0..3 {
        z *= y;
    }
    assert(x % z == 0);
}
```

Generate a `Prover.toml` using `nargo check`.

```
nargo check
```

This command will generate the following `Prover.toml`:

```toml
x = ""
y = ""
```

Edit `Prover.toml` so that it contains an input for which the program will work:

```toml
x = "3072"
y = "4"
```

At this point, you can run `nargo prove` and `nargo verify` to check that the
program is valid using our PLONKY2 backend. You can also execute it via `nargo
execute` or run it in a debugger via `nargo debug` which are both upstream
features.

That said, this page is about `nargo trace`. Using this new command, you can
generate a recording (a trace) of the execution of the program and save that in
a file. In order for this to work, you need to specify a directory where the
recording should be stored via the `--trace-dir` command line parameter.

```
mkdir traces
nargo trace --trace-dir traces
```

The `traces` directory will now contain three files, two of which are metadata.
The main file containing the trace is `trace.json`. You could inspect the files
as is, but they are optimized for minimal whitespace, so you can use another
convenience command we have added to the `nargo` CLI: `nargo format-trace`.

```
nargo format-trace traces/trace.json traces/formatted_trace.json
```

This generates the `traces/formatted_trace.json` file, which at the time of
writing of this document looks like this:

```json
[
  { "Type": { "kind": 30, "lang_type": "None", "specific_info": { "kind": "None" } } },
  { "Path": "<relative-to-this>/src/main.nr" },
  { "Function": { "path_id": 0, "line": 1, "name": "main" } },
  { "Type": { "kind": 7, "lang_type": "u64", "specific_info": { "kind": "None" } } },
  { "VariableName": "x" },
  { "VariableName": "y" },
  { "Call": { "function_id": 0, "args": [
    { "variable_id": 0, "value": { "kind": "Int", "i": 3072, "type_id": 1 } },
    { "variable_id": 1, "value": { "kind": "Int", "i": 4, "type_id": 1 } }
  ] } },
  { "Step": { "path_id": 0, "line": 2 } },
  { "Value": { "variable_id": 0, "value": { "kind": "Int", "i": 3072, "type_id": 1 } } },
  { "Value": { "variable_id": 1, "value": { "kind": "Int", "i": 4, "type_id": 1 } } },
  { "Step": { "path_id": 0, "line": 4 } },
  { "Value": { "variable_id": 0, "value": { "kind": "Int", "i": 3072, "type_id": 1 } } },
  { "Value": { "variable_id": 1, "value": { "kind": "Int", "i": 4, "type_id": 1 } } },
  { "VariableName": "z" },
  { "Value": { "variable_id": 2, "value": { "kind": "Int", "i": 4, "type_id": 1 } } },
  { "Step": { "path_id": 0, "line": 6 } },
  { "Value": { "variable_id": 0, "value": { "kind": "Int", "i": 3072, "type_id": 1 } } },
  { "Value": { "variable_id": 1, "value": { "kind": "Int", "i": 4, "type_id": 1 } } },
  { "Value": { "variable_id": 2, "value": { "kind": "Int", "i": 256, "type_id": 1 } } },
  { "Step": { "path_id": 0, "line": 7 } },
  { "Value": { "variable_id": 0, "value": { "kind": "Int", "i": 3072, "type_id": 1 } } },
  { "Value": { "variable_id": 1, "value": { "kind": "Int", "i": 4, "type_id": 1 } } },
  { "Value": { "variable_id": 2, "value": { "kind": "Int", "i": 256, "type_id": 1 } } }
]
```

Do note that this tool is at an early-alpha stage and the output is likely to
change. That said, this guide should have given you an idea of how it is
supposed to behave.

Specifically, from this trace, we can see that the trace contains all the steps
that are taken during the execution of the program, as well as the contents of
all the variables live at each step. Again, this is an early prototype and we
are in the process of heavily optimizing it, including smart tracking of
variables and reducing unnecessary repetition so that we keep the file size to
a minimum.
