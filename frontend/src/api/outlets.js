import axios from 'axios';
import { API_URL } from '../config';

/**
 * @typedef {Object} Outlet
 * @property {string} id
 * @property {string} name
 * @property {string} [description]
 * @property {string} [audience]
 * @property {string[]} [keywords]
 * @property {string} [section_name]
 * @property {string} [website_url]
 * @property {string} created_at
 * @property {string} updated_at
 * @property {boolean} is_active
 * @property {number} [success_rate]
 * @property {string} [last_used]
 */

/**
 * Fetches all outlets with optional filtering and pagination
 * @param {Object} options - Query options
 * @param {boolean} [options.includeInactive=false] - Whether to include inactive outlets
 * @param {number} [options.limit] - Maximum number of outlets to return
 * @param {number} [options.offset] - Number of outlets to skip (for pagination)
 * @returns {Promise<Outlet[]>} List of outlets
 * @throws {Error} If the API request fails
 */
export const fetchAllOutlets = async (options = {}) => {
  try {
    const { includeInactive = false, limit, offset } = options;
    
    // Build query parameters
    const params = new URLSearchParams();
    if (includeInactive) params.append('include_inactive', 'true');
    if (limit) params.append('limit', limit.toString());
    if (offset) params.append('offset', offset.toString());
    
    const response = await axios.get(`${API_URL}/outlets/get_all_outlets?${params.toString()}`);
    
    if (!response.data) {
      throw new Error('No data received from server');
    }
    
    return response.data;
  } catch (error) {
    console.error('Error fetching outlets:', error);
    throw new Error(
      error.response?.data?.detail || 
      error.message || 
      'Failed to fetch outlets'
    );
  }
};

/**
 * Fetches a single outlet by ID
 * @param {string} outletId - The ID of the outlet to fetch
 * @returns {Promise<Outlet>} The outlet data
 * @throws {Error} If the API request fails
 */
export const fetchOutletById = async (outletId) => {
  try {
    const response = await axios.get(`${API_URL}/outlets/${outletId}`);
    
    if (!response.data) {
      throw new Error('No data received from server');
    }
    
    return response.data;
  } catch (error) {
    console.error('Error fetching outlet:', error);
    throw new Error(
      error.response?.data?.detail || 
      error.message || 
      'Failed to fetch outlet'
    );
  }
}; 