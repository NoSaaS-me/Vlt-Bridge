# Strategic Thinking Tool

Use this prompt when the research agent invokes the "think" tool for strategic reflection.

## Context
The agent is researching: {{ subtopic }}
Current tool calls: {{ tool_calls }} / {{ max_tool_calls }}
Sources found so far: {{ sources_count }}

## Reflection Framework

Consider:

1. **Progress Assessment**
   - What have I learned so far?
   - What key questions remain unanswered?
   - Am I on track or going down rabbit holes?

2. **Search Strategy**
   - What queries should I try next?
   - Should I narrow or broaden my search?
   - Are there authoritative sources I should specifically look for?

3. **Completion Check**
   - Do I have enough quality sources (aim for 3-5)?
   - Have I covered the main aspects of this subtopic?
   - Is there a clear gap I must fill before stopping?

4. **Quality Check**
   - Are my sources diverse (not all from same site)?
   - Do I have recent/current information?
   - Are my sources credible?

## Decision Output

After reflection, decide:
- **CONTINUE**: More searches needed (specify what)
- **STOP**: Sufficient information gathered
