import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'EpochDB',
  description: 'Lossless, Tiered Agentic Memory Engine & Relational Knowledge Subsystem',
  cleanUrls: true,

  head: [
    ['link', { rel: 'icon', href: '/logo.png', type: 'image/png' }],
    ['meta', { name: 'theme-color', content: '#06b6d4' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:title', content: 'EpochDB — Agentic Memory Engine' }],
    ['meta', { property: 'og:description', content: 'Lossless tiered storage, atomic state management, multi-hop retrieval, and neuro-symbolic verification for AI agents.' }],
    ['meta', { property: 'og:image', content: '/logo.png' }],
  ],

  themeConfig: {
    logo: '/logo.png',
    siteTitle: 'EpochDB',

    nav: [
      { text: 'Guide', link: '/guide/introduction' },
      { text: 'Architecture', link: '/guide/architecture' },
      { text: 'API Reference', link: '/api/overview' },
      { text: 'Integrations', link: '/integrations/aster-framework' },
      { text: 'Examples', link: '/examples/basic-usage' },
      {
        text: 'v1.11.0',
        items: [
          { text: 'Changelog', link: '/changelog' },
          { text: 'Benchmarks', link: '/guide/benchmarks' },
          { text: 'GitHub', link: 'https://github.com/jersobh/epochdb' },
        ]
      }
    ],

    sidebar: {
      '/guide/': [
        {
          text: 'Getting Started',
          items: [
            { text: 'Introduction', link: '/guide/introduction' },
            { text: 'Philosophy & Design', link: '/guide/philosophy' },
            { text: 'Comparison vs Mem0 / Letta / Graphiti', link: '/guide/comparison' },
            { text: 'Installation', link: '/guide/installation' },
            { text: 'Quickstart', link: '/guide/quickstart' },
          ]
        },
        {
          text: 'Core Engine',
          items: [
            { text: 'Architecture Overview', link: '/guide/architecture' },
            { text: 'Tiered Storage (Hot & Cold)', link: '/guide/tiered-storage' },
            { text: 'Retrieval Pipeline', link: '/guide/retrieval-pipeline' },
            { text: 'Atomic State & Supersession', link: '/guide/atomic-state-and-supersession' },
            { text: 'Knowledge Graph & GEI', link: '/guide/knowledge-graph' },
            { text: 'Skill Memory', link: '/guide/skill-memory' },
            { text: 'Neuro-Symbolic LCAG', link: '/guide/neuro-symbolic-lcag' },
          ]
        },
        {
          text: 'Advanced Capabilities',
          items: [
            { text: 'Quantitative Logic & SAT', link: '/guide/quantitative-logic' },
            { text: 'DuckDB Cold Analytics', link: '/guide/duckdb-analytics' },
            { text: 'Performance & Benchmarks', link: '/guide/benchmarks' },
          ]
        }
      ],
      '/api/': [
        {
          text: 'API Reference',
          items: [
            { text: 'API Overview', link: '/api/overview' },
            { text: 'EpochDB (Sync Facade)', link: '/api/epochdb' },
            { text: 'AsyncEpochDB (Async Facade)', link: '/api/async-epochdb' },
            { text: 'Remote Client & Server', link: '/api/remote-client' },
            { text: 'Domain Models (Memory, Entity, Graph)', link: '/api/domain-models' },
            { text: 'Symbolic Validators (LCAG)', link: '/api/validators' },
            { text: 'LangGraph Checkpointer', link: '/api/checkpointer' },
            { text: 'VectorStore & Retriever', link: '/api/vectorstore' },
            { text: 'Configuration & Mathematical Constants', link: '/api/config-constants' },
          ]
        }
      ],
      '/integrations/': [
        {
          text: 'Ecosystem Integrations',
          items: [
            { text: 'Aster Framework (OABS)', link: '/integrations/aster-framework' },
            { text: 'LangGraph State Checkpoint', link: '/integrations/langgraph' },
            { text: 'LangChain & LlamaIndex', link: '/integrations/langchain' },
            { text: 'Model Context Protocol (MCP)', link: '/integrations/mcp-server' },
            { text: 'Embedding Providers', link: '/integrations/embedding-providers' },
          ]
        }
      ],
      '/examples/': [
        {
          text: 'Practical Guides & Demos',
          items: [
            { text: 'Basic Memory Operations', link: '/examples/basic-usage' },
            { text: 'Multi-Hop Relational Reasoning', link: '/examples/relational-reasoning' },
            { text: 'Neuro-Symbolic Guardrails', link: '/examples/neuro-symbolic-guardrails' },
            { text: 'Quantitative Cascades', link: '/examples/quantitative-cascades' },
            { text: 'DuckDB SQL Aggregations', link: '/examples/analytics-duckdb' },
            { text: 'Sync vs. Async Benchmark', link: '/examples/sync-vs-async' },
          ]
        }
      ]
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/jersobh/epochdb' }
    ],

    search: {
      provider: 'local'
    },

    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2026 Jefferson (jersobh). Built for Autonomous Agents.'
    },

    editLink: {
      pattern: 'https://github.com/jersobh/epochdb/edit/main/docs/:path',
      text: 'Suggest changes to this page'
    }
  }
})
