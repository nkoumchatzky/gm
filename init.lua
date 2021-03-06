----------------------------------------------------------------------
--
-- Copyright (c) 2012 Clement Farabet
-- 
-- Permission is hereby granted, free of charge, to any person obtaining
-- a copy of this software and associated documentation files (the
-- "Software"), to deal in the Software without restriction, including
-- without limitation the rights to use, copy, modify, merge, publish,
-- distribute, sublicense, and/or sell copies of the Software, and to
-- permit persons to whom the Software is furnished to do so, subject to
-- the following conditions:
-- 
-- The above copyright notice and this permission notice shall be
-- included in all copies or substantial portions of the Software.
-- 
-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
-- EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
-- MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
-- NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
-- LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
-- OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
-- WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
-- 
----------------------------------------------------------------------
-- description:
--     gm - a (super simple) graphical model package for Torch.
--          for now, it only provides means of decoding graphical
--          models (i.e. computing their highest potential state),
--          doing inference, and training CRFs.
--
-- history: 
--     February 2012 - initial draft - Clement Farabet
----------------------------------------------------------------------

require 'xlua'
require 'torch'
require 'dok'

-- package
gm = {}

-- C routines
require 'libgm'

-- extra code
require 'gm.decode'
require 'gm.infer'
require 'gm.energies'
require 'gm.sample'
require 'gm.examples'
require 'gm.adjacency'

----------------------------------------------------------------------
-- creates a graph
--
function gm.graph(...)
   -- usage
   local args, adj, nStates, nodePot, edgePot, typ, maxIter, verbose = dok.unpack(
      {...},
      'gm.graph',
      'create a graphical model from an adjacency matrix',
      {arg='adjacency', type='torch.Tensor | table', help='binary adjacency matrix (N x N tensor, or N-entry sparse table)', req=true},
      {arg='nStates', type='number | torch.Tensor | table', help='number of states per node (N, or a single number)', default=1},
      {arg='nodePot', type='torch.Tensor', help='unary/node potentials (N x nStates)'},
      {arg='edgePot', type='torch.Tensor', help='joint/edge potentials (N x nStates x nStates)'},
      {arg='type', type='string', help='type of graph: crf | mrf | generic', default='generic'},
      {arg='maxIter', type='number', help='maximum nb of iterations for loopy graphs', default=1},
      {arg='verbose', type='boolean', help='verbose mode', default=false}
   )

   -- shortcuts
   local zeros = torch.zeros
   local ones = torch.ones
   local eye = torch.eye
   local Tensor = torch.Tensor
   local sort = torch.sort

   -- graph structure
   local graph = {}

   -- construct list of edges, from adjacency matrix
   local nNodes,nEdges,edgeEnds
   if type(adj) == 'table' then
      nNodes = #adj
      nEdges = 0
      for node1,nodes2 in ipairs(adj) do
         for node2 in pairs(nodes2) do
            nEdges = nEdges + 1
         end
      end
      nEdges = nEdges / 2
      edgeEnds = zeros(nEdges,2)
      local k = 1
      for node1,nodes2 in ipairs(adj) do
         for node2 in pairs(nodes2) do
            if node1 < node2 then
               edgeEnds[k][1] = node1
               edgeEnds[k][2] = node2
               k = k + 1
            end
         end
      end
   else
      nNodes = adj:size(1)
      nEdges = adj:sum()/2
      edgeEnds = zeros(nEdges,2)
      local k = 1
      for i = 1,nNodes do
         for j = 1,nNodes do
            if i < j and adj[i][j] == 1 then
               edgeEnds[k][1] = i
               edgeEnds[k][2] = j
               k = k + 1
            end
         end
      end
   end

   -- count incident edges for each variable
   local nNei = zeros(nNodes)
   --local nei
   if type(adj) == 'table' then
      nei = {}
      for n = 1,nNodes do
         nei[n] = {}
      end
   else
      nei = zeros(nNodes,nNodes)
   end
   for e = 1,nEdges do
      local n1 = edgeEnds[e][1]
      local n2 = edgeEnds[e][2]
      nNei[n1] = nNei[n1] + 1
      nNei[n2] = nNei[n2] + 1
      nei[n1][nNei[n1]] = e
      nei[n2][nNei[n2]] = e
   end

   -- compute (V,E) with V[i] the sum of the nb of edges connected to 
   -- nodes (1,2,...,i-1) plus 1
   -- and E[i] the indexes of nodes connected to node i
   local V = zeros(nNodes+1)
   local E = zeros(nEdges*2)
   local edge = 1
   for n = 1,nNodes do
      V[n] = edge
      if type(nei) == 'table' then
         table.sort(nei[n])
         local nodeEdges = nei[n]
         for i = 1,#nodeEdges do
            E[edge+i-1] = nodeEdges[i]
         end
         edge = edge + #nodeEdges
      else
         local nodeEdges = sort(nei[{ n,{1,nNei[n]} }])
         E[{ {edge, edge+nodeEdges:size(1)-1} }] = nodeEdges
         edge = edge + nodeEdges:size(1)
      end
   end
   V[nNodes+1] = edge

   -- create graph structure
   graph.edgeEnds = edgeEnds
   graph.V = V
   graph.E = E
   graph.nNodes = nNodes
   graph.nEdges = nEdges
   if type(nStates) == 'number' then
      graph.nStates = Tensor(nNodes):fill(nStates)
   elseif type(nStates) == 'table' then
      if #nStates ~= nNodes then
         error('#nStates must be equal to nNodes')
      end
      graph.nStates = Tensor{nStates}
   end
   graph.adjacency = adj
   graph.maxIter = maxIter
   graph.verbose = verbose
   graph.type = args.type
   graph.timer = torch.Timer()

   -- type?
   if graph.type == 'crf' or graph.type == 'mrf' or graph.type == 'generic' then
      -- all good
   else
      xlua.error('unknown graph type: ' .. graph.type, 'gm.graph')
   end

   -- store nodePot/edgePot if given
   graph.nodePot = nodePot
   graph.edgePot = edgePot

   -- some functions
   graph.getEdgesOf = function(g,node)
      return g.E[{ {g.V[node],g.V[node+1]-1} }]
   end

   graph.getNeighborsOf = function(g,node)
      local edges = g:getEdgesOf(node)
      local neighbors = Tensor(edges:size(1))
      local k = 1
      for i = 1,edges:size(1) do
         local edge = g.edgeEnds[edges[i]]
         if edge[1] ~= node then
            neighbors[k] = edge[1]
         else
            neighbors[k] = edge[2]
         end
         k = k + 1
      end
      return neighbors
   end

   graph.setPotentials = function(g,nodePot,edgePot)
      if not nodePot or not edgePot then
         print(xlua.usage('setPotentials',
               'set potentials of an existing graph', nil,
               {type='torch.Tensor', help='unary potentials', req=true},
               {type='torch.Tensor', help='joint potentials', req=true}))
         xlua.error('missing arguments','setPotentials')
      end
      g.nodePot = nodePot
      g.edgePot = edgePot
   end

   graph.decode = function(g,method,maxIter)
      if not method or not gm.decode[method] then
         local availmethods = {}
         for k in pairs(gm.decode) do
            table.insert(availmethods,k)
         end
         availmethods = table.concat(availmethods, ' | ')
         print(xlua.usage('decode',
               'compute optimal state of graph', nil,
               {type='string', help='decoding method: ' .. availmethods, req=true},
               {type='number', help='maximum nb of iterations (used by some methods)', default='graph.maxIter'}))
         xlua.error('missing/incorrect method','decode')
      end
      graph.timer:reset()
      local state = gm.decode[method](g, maxIter or g.maxIter)
      local t = graph.timer:time()
      if g.verbose then
         print('<gm.decode.'..method..'> decoded graph in ' .. t.real .. 'sec')
      end
      return state
   end

   graph.infer = function(g,method,maxIter)
      if not method or not gm.infer[method] then
         local availmethods = {}
         for k in pairs(gm.infer) do
            table.insert(availmethods,k)
         end
         availmethods = table.concat(availmethods, ' | ')
         print(xlua.usage('infer',
               'compute optimal state of graph', nil,
               {type='string', help='inference method: ' .. availmethods, req=true},
               {type='number', help='maximum nb of iterations (used by some methods)', default='graph.maxIter'}))
         xlua.error('missing/incorrect method','infer')
      end
      graph.timer:reset()
      local nodeBel,edgeBel,logZ = gm.infer[method](g, maxIter or g.maxIter)
      local t = graph.timer:time()
      if g.verbose then
         print('<gm.infer.'..method..'> performed inference on graph in ' .. t.real .. 'sec')
      end
      return nodeBel,edgeBel,logZ
   end

   graph.sample = function(g,method,n)
      if not method or not gm.sample[method] then
         local availmethods = {}
         for k in pairs(gm.sample) do
            table.insert(availmethods,k)
         end
         availmethods = table.concat(availmethods, ' | ')
         print(xlua.usage('sample',
               'sample from model', nil,
               {type='string', help='sampling method: ' .. availmethods, req=true},
               {type='number', help='nb of samples', default=1}))
         xlua.error('missing/incorrect method','infer')
      end
      graph.timer:reset()
      local samples = gm.sample[method](g, n or 1)
      local t = graph.timer:time()
      if g.verbose then
         print('<gm.sample.'..method..'> performed sampling from graph in ' .. t.real .. 'sec')
      end
      if not n then return samples[1] end
      return samples
   end

   graph.initParameters = function(g,nodeMap,edgeMap)
      if not nodeMap or not edgeMap then
         print(xlua.usage('initParameters',
               'init trainable parameters (for crf/mrf graphs)', nil,
               {type='torch.Tensor', help='map from node potentials to parameters', req=true},
               {type='torch.Tensor', help='map from edge potentials to parameters', req=true}))
         xlua.error('missing arguments','initParameters')
      end
      g.nodeMap = nodeMap
      g.edgeMap = edgeMap
      g.nParams = math.max(nodeMap:max(),edgeMap:max())
      g.w = zeros(g.nParams)
   end

   graph.makePotentials = function(g,...)
      local Xnode,Xedge
      local args = {...}
      if g.type == 'crf' then
         Xnode = args[1]
         Xedge = args[2]
      end
      if not g.w then
         xlua.error('graph doesnt have parameters, call g:initParameters() first','makePotentials')
      end
      if (g.type == 'crf' and not Xnode) then
         print(xlua.usage('makePotentials',
               'make potentials from internal parameters (for crf graphs) and given node/edge features', nil,
               {type='torch.Tensor', help='node features', req=true},
               {type='torch.Tensor', help='edge features', req=true}))
         xlua.error('missing arguments / incorrect graph','makePotentials')
      end
      gm.energies[g.type].makePotentials(g,g.w,g.nodeMap,g.edgeMap,Xnode,Xedge)
   end

   graph.nll = function(g,method,Y,Xnode,Xedge)
      if not g.w then
         xlua.error('graph doesnt have parameters, call g:initParameters() first','nll')
      end
      if not Y or not method or not gm.infer[method] or not gm.energies[g.type] then
         local availmethods = {}
         for k in pairs(gm.infer) do
            table.insert(availmethods,k)
         end
         availmethods = table.concat(availmethods, ' | ')
         if g.type == 'crf' then
            print(xlua.usage('nll',
               'compute negative log-likelihood of CRF, and its gradient wrt weights', nil,
               {type='string', help='inference method: ' .. availmethods, req=true},
               {type='torch.Tensor', help='labeling', req=true},
               {type='torch.Tensor', help='node features', req=true},
               {type='torch.Tensor', help='edge features', req=true}
               ))
         elseif g.type == 'mrf' then
            print(xlua.usage('nll',
               'compute negative log-likelihood of MRF, and its gradient wrt weights', nil,
               {type='string', help='inference method: ' .. availmethods, req=true},
               {type='torch.Tensor', help='node values', req=true}
               ))
         end
         xlua.error('missing/incorrect arguments / incorrect graph','nll')
      end
      graph.timer:reset()
      local f,grad
      if g.type == 'crf' then
         f,grad = gm.energies[g.type].nll(g, g.w,
                                       g.nodeMap,g.edgeMap, method,
                                       g.maxIter,
                                       Y, Xnode, Xedge)
      elseif g.type == 'mrf' then
         f,grad = gm.energies[g.type].nll(g, g.w,
                                       g.nodeMap,g.edgeMap, method,
                                       g.maxIter,
                                       Y)
      end
      local t = graph.timer:time()
      if g.verbose then
         print('<gm.nll.'..method..'> computed negative log-likelihood in ' .. t.real .. 'sec')
      end
      return f,grad
   end

   graph.getPotentialForConfig = function(g,y)
      if not y then
         print(xlua.usage('getPotentialForConfig',
               'get potential for a given configuration', nil,
               {type='torch.Tensor', help='configuration of all nodes in graph', req=true}))
         xlua.error('missing config','getPotentialForConfig')
      end
      -- return potential
      return g.nodePot.gm.getPotentialForConfig(g.nodePot,g.edgePot,g.edgeEnds,y)
   end

   graph.getLogPotentialForConfig = function(g,y)
      if not y then
         print(xlua.usage('getLogPotentialForConfig',
               'get log potential for a given configuration', nil,
               {type='torch.Tensor', help='configuration of all nodes in graph', req=true}))
         xlua.error('missing config','getPotentialForConfig')
      end
      -- return potential
      return g.nodePot.gm.getLogPotentialForConfig(g.nodePot,g.edgePot,g.edgeEnds,y)
   end

   local tostring = function(g)
      local str = 'gm.GraphicalModel\n'
      str = str .. ' + nb of nodes: ' .. g.nNodes .. '\n'
      str = str .. ' + nb of edges: ' .. g.nEdges .. '\n'
      str = str .. ' + maximum nb of states per node: ' .. g.nStates:max()
      return str
   end
   setmetatable(graph, {__tostring=tostring})

   -- verbose?
   if graph.verbose then
      print('<gm.graph> created new graphical model:')
      print(tostring(graph))
   end

   -- return result
   return graph
end

-- Return package
return gm
