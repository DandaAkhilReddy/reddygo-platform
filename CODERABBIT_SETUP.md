# CodeRabbit Setup Instructions

CodeRabbit is an AI-powered code review tool that automatically reviews your pull requests.

## ✅ What's Already Configured

- ✅ `.coderabbit.yaml` - Configuration file
- ✅ GitHub Secret `CODERABBIT_API_KEY` - API key stored securely
- ✅ Test PR Created: #1 (Friends System)

## 🚀 How to Enable CodeRabbit Reviews

CodeRabbit works as a **GitHub App**, not a GitHub Action. Follow these steps:

### Option 1: Install CodeRabbit GitHub App (Recommended)

1. **Visit CodeRabbit**
   - Go to https://coderabbit.ai
   - Click "Sign in with GitHub"

2. **Install the App**
   - Select your repository: `DandaAkhilReddy/reddygo-platform`
   - Grant permissions for:
     - Read access to code
     - Write access to pull requests and issues

3. **Verify Installation**
   - Open PR #1: https://github.com/DandaAkhilReddy/reddygo-platform/pull/1
   - CodeRabbit should automatically post a review within 1-2 minutes

### Option 2: Use API Key Directly (Advanced)

If you prefer API integration:

1. **API Key Already Set**: Your key `cr-c7b7c2da54cf5a77d3b50747dffc104fe51cf625df227cccbb5e272f96` is stored as a GitHub secret

2. **Manual API Calls**: Use curl or Python to trigger reviews:

```bash
curl -X POST https://api.coderabbit.ai/v1/reviews \
  -H "Authorization: Bearer $CODERABBIT_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "repo": "DandaAkhilReddy/reddygo-platform",
    "pr_number": 1
  }'
```

## 📋 CodeRabbit Configuration

Our `.coderabbit.yaml` settings:

- **Profile**: `chill` - Less strict, focuses on major issues
- **Auto Review**: Enabled for all PRs (except WIP/DO NOT REVIEW)
- **High-level Summary**: Enabled
- **Review Status**: Shows review progress
- **Shellcheck**: Enabled for bash scripts
- **Auto Reply**: Responds to review comments

## 🧪 Test the Setup

We've created a test PR with the Friends System feature:

**PR #1**: Friends System - Social Features & Friend Leaderboards
- URL: https://github.com/DandaAkhilReddy/reddygo-platform/pull/1
- Code: ~450 lines of Python (FastAPI router)
- Features: Friend requests, bidirectional friendships, real-time leaderboards

**Expected CodeRabbit Review:**
- Code quality suggestions
- Security best practices
- Performance optimizations
- Python/FastAPI specific recommendations

## 🔧 Troubleshooting

### CodeRabbit Not Reviewing?

1. **Check Installation**
   ```bash
   gh api repos/DandaAkhilReddy/reddygo-platform/installation
   ```

2. **Verify Permissions**
   - Go to https://github.com/settings/installations
   - Click "CodeRabbit"
   - Ensure `reddygo-platform` is selected

3. **Check PR Comments**
   - CodeRabbit posts reviews as PR comments
   - Look for comments from `@coderabbitai` bot

### Still Not Working?

- Re-install the app
- Check GitHub App permissions
- Verify repository access in CodeRabbit dashboard

## 📚 Learn More

- **CodeRabbit Docs**: https://docs.coderabbit.ai
- **GitHub App**: https://github.com/apps/coderabbitai
- **Pricing**: Free for open-source, paid for private repos

---

## Next Steps After CodeRabbit Review

Once CodeRabbit reviews PR #1, you can:

1. **Review Suggestions** - Check CodeRabbit's comments
2. **Make Improvements** - Address high-priority issues
3. **Merge PR** - Merge the Friends System to master
4. **Continue Development** - Create more features for review

CodeRabbit will automatically review all future PRs! 🚀
