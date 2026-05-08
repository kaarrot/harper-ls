# `harper-ls`

Documentation for `harper-ls` has moved to the main [website](https://writewithharper.com/docs/integrations/language-server).

## Completion Configuration

`harper-ls` accepts completion settings under the `harper-ls.completion` object.

```json
{
  "harper-ls": {
    "completion": {
      "enabled": true,
      "minPrefixLength": 2,
      "maxResults": 7,
      "commitWithSpace": "confident"
    }
  }
}
```

`commitWithSpace` controls whether typing a space accepts completion items:

- `"never"`: space always inserts a space; completions require explicit selection.
- `"confident"`: default. Space accepts only the top completion when Harper is confident it is clearly better than the alternatives.
- `"always"`: space accepts completion items whenever the client supports commit characters.
