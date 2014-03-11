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
--     gm.adjacency - a list of functions to create adjacency matrices
--
-- history: 
--     February 2012 - initial draft - Clement Farabet
----------------------------------------------------------------------

-- that table contains standard functions to create adjacency matrices
gm.adjacency = {}

-- shortcuts
local zeros = torch.zeros
local ones = torch.ones
local eye = torch.eye
local sort = torch.sort
local log = torch.log
local exp = torch.exp

-- messages
local warning = function(msg)
   print(sys.COLORS.red .. msg .. sys.COLORS.none)
end

----------------------------------------------------------------------
-- Full graph adjacency
--
function gm.adjacency.full(nNodes)
   return ones(nNodes,nNodes) - eye(nNodes)
end

----------------------------------------------------------------------
-- Builds a satellite adjacency matrix : fully connected suns nodes with satellites nodes connected to all the sun nodes 
-- Meant for a directed graph since the suns points towards the satellites and not the reverse
--
function gm.adjacency.solar(nSuns, nSatellites)
   local nNodes = nSatellites + nSuns
   local adj = ones(nNodes,nNodes) - eye(nNodes)
  
   -- Satellites connections
   for i = nSuns+1,nNodes do
      for j = nSuns+1,nNodes do
         adj[i][j] = 0
      end
   end
   return adj         
end
----------------------------------------------------------------------
-- N-connexity 2D lattice (N = 4 or 8)
--
function gm.adjacency.lattice2d(nRows,nCols,connex)
   local nNodes = nRows*nCols
   local adj = {}
   for n = 1,nNodes do
      adj[n] = {}
   end
   if connex == 4 then
      for i = 1,nRows do
         for j = 1,nCols do
            local n = (i-1)*nCols + j
            if j < nCols then
               adj[n][n+1] = 1
               adj[n+1][n] = 1
            end
            if i < nRows then
               adj[n][n+nCols] = 1
               adj[n+nCols][n] = 1
            end
         end
      end
   elseif connex == 8 then
      for i = 1,nRows do
         for j = 1,nCols do
            local n = (i-1)*nCols + j
            if j < nCols then
               adj[n][n+1] = 1
               adj[n+1][n] = 1
            end
            if i < nRows then
               adj[n][n+nCols] = 1
               adj[n+nCols][n] = 1
            end
            if i < nRows and j < nCols then
               adj[n][n+nCols+1] = 1
               adj[n+nCols+1][n] = 1
            end
            if i < nRows and j > 1 then
               adj[n][n+nCols-1] = 1
               adj[n+nCols-1][n] = 1
            end
         end
      end
   else
      sys.error('connexity can only be 4 or 8 on a 2D lattice', 'gm.adjacency.lattice2d')
   end
   return adj
end

function gm.adjacency.listOfEdges(adj)
   -- construct list of edges, from adjacency matrix
   local nNodes,nEdges,edgeEnds
   if type(adj) == 'table' then
      nNodes = #adj      
      nEdges = 0
      print(ipairs(adj))
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
   return nNodes,nEdges,edgeEnds
end
----------------------------------------------------------------------
-- N-connexity 3D lattice (N = 6 or 26)
--
function gm.adjacency.lattice3d(nRows,nCols,nLayers,connex)
   sys.error('not implemented yet', 'gm.adjacency.lattice3d')
end
